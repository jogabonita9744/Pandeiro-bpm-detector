package com.pandeiro.bpm;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.SeekBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.util.ArrayList;
import java.util.Collections;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST = 1;
    private static final int SAMPLE_RATE = 44100;
    private static final int BUFFER_SIZE_FACTOR = 4;

    private AudioRecord audioRecord;
    private Thread recordingThread;
    private volatile boolean isRecording = false;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    // Beat detection state
    private float[] prevSpectrum = null;
    private final ArrayList<Float> fluxHistory = new ArrayList<>();
    private long lastBeatTime = 0;
    private final ArrayList<Long> beatTimes = new ArrayList<>();
    private int sensitivity = 3; // 1-5

    // UI
    private TextView bpmText;
    private TextView tempoText;
    private TextView statusText;
    private Button startStopBtn;
    private View pandeiroPulse;
    private SeekBar sensBar;
    private TextView sensVal;

    // Simple FFT (Cooley-Tukey)
    private static final int FFT_SIZE = 2048;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        bpmText     = findViewById(R.id.bpmText);
        tempoText   = findViewById(R.id.tempoText);
        statusText  = findViewById(R.id.statusText);
        startStopBtn= findViewById(R.id.startStopBtn);
        pandeiroPulse = findViewById(R.id.pandeiroPulse);
        sensBar     = findViewById(R.id.sensBar);
        sensVal     = findViewById(R.id.sensVal);

        sensBar.setMax(4);
        sensBar.setProgress(2); // default sensitivity 3
        sensBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar s, int p, boolean u) {
                sensitivity = p + 1;
                sensVal.setText(String.valueOf(sensitivity));
            }
            public void onStartTrackingTouch(SeekBar s) {}
            public void onStopTrackingTouch(SeekBar s) {}
        });

        startStopBtn.setOnClickListener(v -> {
            if (isRecording) stopRecording();
            else checkPermissionAndStart();
        });

        findViewById(R.id.resetBtn).setOnClickListener(v -> resetBPM());
    }

    private void checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        } else {
            ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.RECORD_AUDIO}, PERMISSION_REQUEST);
        }
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] perms, int[] results) {
        super.onRequestPermissionsResult(req, perms, results);
        if (req == PERMISSION_REQUEST && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        } else {
            statusText.setText("Permission micro refusée");
        }
    }

    private void startRecording() {
        int bufSize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) * BUFFER_SIZE_FACTOR;

        audioRecord = new AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize
        );

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            statusText.setText("Erreur initialisation micro");
            return;
        }

        isRecording = true;
        audioRecord.startRecording();
        startStopBtn.setText("ARRÊTER");
        startStopBtn.setBackgroundColor(0xFFD63B2F);
        setStatus("● Écoute en cours…", true);

        prevSpectrum = null;
        fluxHistory.clear();
        lastBeatTime = 0;

        final int hopSize = FFT_SIZE / 2;
        final short[] buffer = new short[hopSize];
        final float[] window = makeHannWindow(FFT_SIZE);
        final float[] frame = new float[FFT_SIZE];
        final float[] overlap = new float[FFT_SIZE - hopSize];

        recordingThread = new Thread(() -> {
            while (isRecording) {
                int read = audioRecord.read(buffer, 0, hopSize);
                if (read <= 0) continue;

                // Build frame with overlap
                System.arraycopy(overlap, 0, frame, 0, FFT_SIZE - hopSize);
                for (int i = 0; i < hopSize; i++)
                    frame[FFT_SIZE - hopSize + i] = buffer[i] / 32768f;
                System.arraycopy(frame, hopSize, overlap, 0, FFT_SIZE - hopSize);

                // Apply Hann window
                float[] windowed = new float[FFT_SIZE];
                for (int i = 0; i < FFT_SIZE; i++)
                    windowed[i] = frame[i] * window[i];

                // FFT
                float[] re = windowed.clone();
                float[] im = new float[FFT_SIZE];
                fft(re, im);

                // Magnitude spectrum (lower 25%)
                int binEnd = FFT_SIZE / 4 / 2; // lower 25% of positive freqs
                float[] mag = new float[binEnd];
                float maxMag = 0;
                for (int i = 0; i < binEnd; i++) {
                    mag[i] = (float) Math.sqrt(re[i]*re[i] + im[i]*im[i]);
                    if (mag[i] > maxMag) maxMag = mag[i];
                }
                // Normalize
                if (maxMag > 0)
                    for (int i = 0; i < binEnd; i++) mag[i] /= maxMag;

                // Spectral flux
                float flux = 0;
                if (prevSpectrum != null) {
                    for (int i = 0; i < binEnd; i++) {
                        float d = mag[i] - prevSpectrum[i];
                        if (d > 0) flux += d;
                    }
                }
                prevSpectrum = mag.clone();

                // Adaptive threshold
                fluxHistory.add(flux);
                if (fluxHistory.size() > 43) fluxHistory.remove(0);

                float mean = 0;
                for (float f : fluxHistory) mean += f;
                mean /= fluxHistory.size();

                float variance = 0;
                for (float f : fluxHistory) variance += (f - mean) * (f - mean);
                float stdev = (float) Math.sqrt(variance / fluxHistory.size());

                float mult = 3.5f - sensitivity * 0.5f;
                float thresh = mean + mult * stdev;

                long now = System.currentTimeMillis();
                long minGap = 230 - sensitivity * 20L;

                if (flux > thresh && flux > 0.1f && (now - lastBeatTime) > minGap) {
                    lastBeatTime = now;
                    onBeat(now);
                }
            }
        });
        recordingThread.start();
    }

    private void onBeat(long now) {
        beatTimes.add(now);
        if (beatTimes.size() > 24) beatTimes.remove(0);

        Integer bpm = calcBPM();

        mainHandler.post(() -> {
            // Pulse animation
            pandeiroPulse.animate().scaleX(1.08f).scaleY(1.08f).setDuration(60)
                .withEndAction(() -> pandeiroPulse.animate().scaleX(1f).scaleY(1f).setDuration(100).start())
                .start();

            if (bpm != null) {
                bpmText.setText(String.valueOf(bpm));
                tempoText.setText(tempoName(bpm));
            }
            setStatus("● Coup détecté", true);
        });
    }

    private Integer calcBPM() {
        if (beatTimes.size() < 2) return null;
        ArrayList<Long> intervals = new ArrayList<>();
        for (int i = 1; i < beatTimes.size(); i++)
            intervals.add(beatTimes.get(i) - beatTimes.get(i-1));
        Collections.sort(intervals);
        long med = intervals.get(intervals.size() / 2);
        ArrayList<Long> clean = new ArrayList<>();
        for (long iv : intervals)
            if (iv > med * 0.5 && iv < med * 2.5) clean.add(iv);
        if (clean.isEmpty()) return null;
        long sum = 0;
        for (long iv : clean) sum += iv;
        long avg = sum / clean.size();
        int bpm = (int) Math.round(60000.0 / avg);
        return (bpm >= 20 && bpm <= 400) ? bpm : null;
    }

    private void stopRecording() {
        isRecording = false;
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }
        startStopBtn.setText("DÉMARRER");
        startStopBtn.setBackgroundColor(0xFFF0B429);
        setStatus("Arrêté", false);
    }

    private void resetBPM() {
        beatTimes.clear();
        fluxHistory.clear();
        prevSpectrum = null;
        bpmText.setText("--");
        tempoText.setText("");
        setStatus(isRecording ? "● Écoute en cours…" : "En attente", isRecording);
    }

    private void setStatus(String msg, boolean active) {
        mainHandler.post(() -> {
            statusText.setText(msg);
            statusText.setTextColor(active ? 0xFFD63B2F : 0xFF7A4E22);
        });
    }

    private String tempoName(int bpm) {
        if (bpm < 60)  return "Largo";
        if (bpm < 66)  return "Larghetto";
        if (bpm < 76)  return "Adagio";
        if (bpm < 108) return "Andante";
        if (bpm < 120) return "Moderato";
        if (bpm < 156) return "Allegro";
        if (bpm < 176) return "Vivace";
        if (bpm < 200) return "Presto";
        return "Prestissimo";
    }

    // ── Hann window ──────────────────────────────────────────────
    private float[] makeHannWindow(int size) {
        float[] w = new float[size];
        for (int i = 0; i < size; i++)
            w[i] = (float)(0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1))));
        return w;
    }

    // ── In-place Cooley-Tukey FFT ────────────────────────────────
    private void fft(float[] re, float[] im) {
        int n = re.length;
        // Bit reversal
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                float t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
            }
        }
        // FFT
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2 * Math.PI / len;
            float wRe = (float) Math.cos(ang);
            float wIm = (float) Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                float curRe = 1, curIm = 0;
                for (int j = 0; j < len / 2; j++) {
                    float uRe = re[i+j], uIm = im[i+j];
                    float vRe = re[i+j+len/2]*curRe - im[i+j+len/2]*curIm;
                    float vIm = re[i+j+len/2]*curIm + im[i+j+len/2]*curRe;
                    re[i+j] = uRe+vRe; im[i+j] = uIm+vIm;
                    re[i+j+len/2] = uRe-vRe; im[i+j+len/2] = uIm-vIm;
                    float newCurRe = curRe*wRe - curIm*wIm;
                    curIm = curRe*wIm + curIm*wRe;
                    curRe = newCurRe;
                }
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopRecording();
    }
}

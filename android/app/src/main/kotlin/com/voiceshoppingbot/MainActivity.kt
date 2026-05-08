package com.voiceshoppingbot

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.launch
import java.util.Locale

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var tts: TextToSpeech
    private lateinit var backend: BackendApiClient

    private lateinit var btnMic: FloatingActionButton
    private lateinit var tvStatus: TextView
    private lateinit var tvTranscript: TextView
    private lateinit var tvResponse: TextView

    private var isListening = false
    private var ttsReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnMic = findViewById(R.id.btnMic)
        tvStatus = findViewById(R.id.tvStatus)
        tvTranscript = findViewById(R.id.tvTranscript)
        tvResponse = findViewById(R.id.tvResponse)

        tts = TextToSpeech(this, this)
        backend = BackendApiClient()

        setupSpeechRecognizer()
        btnMic.setOnClickListener { onMicTapped() }
    }

    private fun onMicTapped() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC
            )
            return
        }
        toggleListening()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_MIC &&
            grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            toggleListening()
        }
    }

    private fun setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) = setStatus("Listening…")
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() = setStatus("Processing…")

            override fun onError(error: Int) {
                isListening = false
                updateMicState()
                setStatus("Tap mic to speak")
            }

            override fun onResults(results: Bundle?) {
                isListening = false
                updateMicState()
                val text = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull() ?: return
                tvTranscript.text = "You: $text"
                askBackend(text)
            }

            override fun onPartialResults(partial: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun toggleListening() {
        if (isListening) {
            speechRecognizer.stopListening()
            isListening = false
        } else {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }
            speechRecognizer.startListening(intent)
            isListening = true
        }
        updateMicState()
    }

    private fun askBackend(message: String) {
        setStatus("Thinking…")
        lifecycleScope.launch {
            try {
                val reply = backend.sendMessage(message)
                tvResponse.text = "Bot: $reply"
                speak(reply)
            } catch (e: Exception) {
                setStatus("Error — try again")
                tvResponse.text = "Error: ${e.message}"
            }
        }
    }

    private fun speak(text: String) {
        if (!ttsReady) return
        setStatus("Speaking…")
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, TTS_UTTERANCE_ID)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.getDefault()
            ttsReady = true
        }
        setStatus("Tap mic to speak")
    }

    private fun updateMicState() {
        val color = if (isListening) {
            getColor(R.color.mic_active)
        } else {
            getColor(R.color.mic_idle)
        }
        btnMic.backgroundTintList = android.content.res.ColorStateList.valueOf(color)
    }

    private fun setStatus(text: String) {
        runOnUiThread { tvStatus.text = text }
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy()
        tts.shutdown()
        backend.shutdown()
    }

    companion object {
        private const val REQ_MIC = 100
        private const val TTS_UTTERANCE_ID = "bot_response"
    }
}

from transformers import pipeline
transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-tiny")
output = transcriber('./src/1.wav')['text']
print(output)

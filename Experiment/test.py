import pyaudio, wave
wf = wave.open(r"C:\Users\barak\Documents\GitHub\startegicMW_lite\Experiment_Natalia\resources\audioFiles\expAudio\segm_0.wav", 'rb')
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                frames_per_buffer=1024)
data = wf.readframes(1024)
while data:
    stream.write(data)
    data = wf.readframes(1024)
stream.stop_stream(); stream.close(); p.terminate()
print("Done")
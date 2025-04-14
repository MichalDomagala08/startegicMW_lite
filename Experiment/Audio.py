import subprocess,sys
subprocess.check_call([sys.executable, "-m","pip","install","pyaudio"])

import pyaudio
import pygame
import wave
import threading
import time
from random import uniform
import sys,os

class audioTrial:

    def __init__(self, audioPath,storyTimeDict, font, screen,currentStage,nextStage, output_control,chunk_size=256,verbose=0):

        ### Screen
        self.font = font
        self.crossFont = pygame.font.Font(None, 100)

        self.screen = screen
        self.width = screen.get_size()[0]
        self.height = screen.get_size()[1]

        ### Audio Stream
        self.audio_folder = audioPath
        self.audio_files = self.load_audio_files()
        self.p = pyaudio.PyAudio()
        self.currentStoryPart = 0; # track which story segment is playing
        self.chunk_size = chunk_size
        self.stream = None  # Stream will be opened dynamically
        self.actual_duration = None # The Duration of Playing of a song will be stored Here 

        ### Audio Contoll Variables
        self.paused_position = 0        # Position in which we paused
        self.audio_playing = False      # Flag of Playing Auduo
        self.paused_for_input = False   # Flag of Pausing for input
        self.initialTime = 0            # intial time of starting our Audio
        self.pausedTime = 0             # Experiment Time in which we paused our stream
        self.nextProbeTime = 0          # Timing of a next probe
        self.initFlag = True            # Flag for initialization
        self.trialFlag = True           # Flag For measuring trials

        ### Debug and Return variables
        self.timingLog = []
        self.responsesTiming = []
        self.verbose = verbose
        self.output_control = output_control;
        sys.stdout = self.output_control  # Redirect standard output

        ### Staging and Experiment Mode:
        self.currentStage = currentStage;
        self.nextStage = nextStage #What Kind of stage do you expect next  
        self.storyParts = storyTimeDict["partNames"]

    def load_audio_files(self):
            """ Load all WAV file as a path from the folder and sort them numerically. """
            files = [f for f in os.listdir(self.audio_folder) if f.endswith(".wav")]
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # Sort by segment number
            return [os.path.join(self.audio_folder, f) for f in files]
    
    def log_keypress(self,key):
            
            if key == pygame.K_1:
                keyname = "1"
            elif key == pygame.K_2:
                keyname = "2"
            elif key == pygame.K_3:
                keyname = "3"
            elif key == pygame.K_4:
                keyname = "4"
            else:
                keyname = "d"

                self.output_control.write(f"WARNING SOMETHING ELSE WAS PRESSED {key}")
            if self.verbose:
                self.output_control.write(f"   Key {key} pressed at: {self.pausedTime - self.initialTime:.3f}; Reaction Time: {(self.pausedTime - self.probeOnset):.3f}")

            self.output_control.writeToEyeLink(message=f"\tKEYPRESS\t{key}\t0")              # Write KeyPress to Eyelink                  
            self.timingLog.append(["KEYPRESS",self.pausedTime  - self.initialTime ,(self.pausedTime - self.probeOnset),self.actual_duration])
            self.responsesTiming.append((keyname , self.pausedTime  - self.initialTime))
            
    def log_part_beg(self):
        self.timingLog.append(["BEGIN",(time.perf_counter() - self.initialTime),0,0])              
        self.responsesTiming.append(("BEGIN", (time.perf_counter() - self.initialTime)))

        message = f"\n   Beginning Part: {self.storyParts[self.currentStoryPart]:12} at exp Time {time.perf_counter()- self.initialTime:9.3f}";
        self.output_control.write(message)

        self.output_control.writeToEyeLink(message=f"\tPART\t{self.storyParts[self.currentStoryPart]:12}\tBEG")

    def log_part_end(self):
        self.timingLog.append(["PROBE",(time.perf_counter() - self.initialTime),0,self.actual_duration])              
        self.responsesTiming.append(("PROBE", (time.perf_counter() - self.initialTime)))

        message = f"   Part ended:     {self.storyParts[self.currentStoryPart]:12} at exp Time {time.perf_counter()- self.initialTime:9.3f} (Duration: {self.actual_duration:.3f})"
        self.output_control.write(message)

        self.output_control.writeToEyeLink(message=f"\tPART\t{self.storyParts[self.currentStoryPart]:12}\tEND")
        # Write to Psychopy
    def display_fixation_cross(self):
        """
        Displays a '+' symbol in the center of the screen.
        """
        self.screen.fill((255 / 2, 255 / 2, 255 / 2))  # Gray background
        text_surface = self.crossFont.render("+", True, (255, 255, 255))  # White "+"
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def display_thought_probe(self,line_height=50):
        """
            Render multi-line text with dynamic left margin (in pixels) and special centering for lines with '# #'.
        """
        # Determine the width of the longest line in pixels
        max_line_pixel_width = max(self.font.size(line.replace("#", "").strip())[0] for line in Question)

        # Calculate the dynamic left margin as a proportion of the unused space
        margin = (self.width - max_line_pixel_width) // 2

        # Calculate the Y position to center the entire block vertically
        total_text_height = len(Question) * line_height
        y_offset = (self.height - total_text_height) // 2

        for i, line in enumerate(Question):
            # Check if the line should be centered (contains '# #')
            if line.startswith("#") and line.endswith("#"):
                clean_line = line.replace("#", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.width // 2, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)

            elif line.startswith("%") and line.endswith("%"):
                clean_line = line.replace("%", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(topleft=(self.width // 3, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)
            else:
                # Left-aligned text at the dynamic margin
                text_surface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (margin, y_offset + i * line_height))
        pygame.display.flip()

    def play_audio(self):
        """
        Starts/resumes playing the audio from each file
        """

        self.audio_playing = True
        self.paused_for_input = False  # Reset for next thought probe

        file_path = self.audio_files[self.currentStoryPart]
        self.audio_file = wave.open(file_path, 'rb')

        if self.stream:
            self.stream.close()  # Close previous stream if it exists

        # Open Stream for playing
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.audio_file.getsampwidth()),
            channels=self.audio_file.getnchannels(),
            rate=self.audio_file.getframerate(),
            output=True
        )
        self.log_part_beg()

        def stream_audio():
            start_time = time.perf_counter()  # Capture the actual start time
            while self.audio_playing:
                data = self.audio_file.readframes(self.chunk_size)
                if data == b'':  # End of file
                    self.audio_playing = False
                    break
                self.stream.write(data)
                # Track the current position (in samples)

            self.audio_playing = False
            self.actual_duration = time.perf_counter() - start_time  # Calculate actual time taken
            self.output_control.write(f"Actual playback duration: {self.actual_duration:.3f} sec")

        # Lof that the audio is beginning:

        # Play audio on a separate thread
        threading.Thread(target=stream_audio, daemon=True).start()

    
    def run(self):
        """
        Runs the core of our audio trial.
        """
        if self.initFlag: # Initailise Experiment Begin as well as first Time to Pause
            self.initialTime = time.perf_counter()  # Start the timer
            self.pausedTime = self.initialTime
            self.initFlag = False;
        
        if self.trialFlag: # Startingto play The first sound at the beggining 
            self.play_audio()
            self.trialFlag = False

        if self.paused_for_input:
            self.display_thought_probe()
        else:
            self.display_fixation_cross()

        # Check if the audio has finished
        if not self.audio_playing:
            if not self.paused_for_input:
                self.probeOnset =  time.perf_counter();
                self.log_part_end()
            self.paused_for_input = True

        ### Waiting for KeyPress on Thought Probe
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.stream.stop_stream()
                self.stream.close()
                return "exit"

            if self.paused_for_input and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1 or event.key == pygame.K_2 or event.key == pygame.K_3 or event.key == pygame.K_4:
              
                    self.paused_for_input = False
                    self.pausedTime = time.perf_counter()  # Reset timer
                    currentKey = event.key
                    self.log_keypress(currentKey) # Logs KeyPresses
                      
                    self.currentStoryPart += 1
                    self.trialFlag = True
                    self.audio_playing = True




        if self.currentStoryPart >= len(self.audio_files):
            self.output_control.write("\nStory Ends.")
            return self.nextStage  # No more segments → End experiment

        return  self.currentStage # Usually return story1 if audio has not ended or we hadnt exit

# A list of Timings of story 1: Used to Clock the Onset of Each Fragment: (in seconds)
storyTimeDict2 = data = {
    'partNames': [ 'ADAM_1', 'MARIA_1', 'ADAM_2', 'MARIA_2', 'ADAM_3', 'MARIA_3', 'ADAM_4', 'MARIA_4', 'ADAM_5', 'MARIA_5', 'ADAM_6', 'MARIA_6', 'ADAM_7', 
                  'MARIA_7', 'ADAM_8', 'MARIA_8', 'ADAM_9', 'MARIA_9', 'ADAM_10', 'MARIA_10', 'ADAM_11', 'MARIA_11'],
    'partTimes': [ 56.678, 56.688, 64.743, 59.355, 58.085, 56.379, 64.487, 62.353, 
                  61.563, 57.115, 59.493, 57.019, 56.283, 59.941, 57.679, 58.043, 59.707, 60.827, 59.663, 59.109, 60.443, 58.832, ]
}

storyTimeDict1 = {
    'partNames': [ 'KASIA_1', 'JANEK_1', 'KASIA_2', 'JANEK_2', 'KASIA_3', 'JANEK_3', 'KASIA_4', 'JANEK_4', 'KASIA_5', 'JANEK_5', 'KASIA_6', 'JANEK_6', 'KASIA_7', 
                  'JANEK_7', 'KASIA_8', 'JANEK_8', 'KASIA_9', 'JANEK_9', 'KASIA_10', 'JANEK_10',  'KASIA_11', 'JANEK_11'],
    'partTimes': [ 61.296, 59.525, 56.655, 59.92, 65.147, 57.574, 61.99, 65.329, 60.965, 
                  60.454, 57.947, 64.176, 60.955, 60.241, 59.472, 57.115, 59.387, 57.306, 58.693, 62.267, 58.063, 61.295, ]
}
storyTimeDict2c = data = {
    'partNames': ['INTRODUCTION', 'ADAM_1', 'MARIA_1', 'ADAM_2', 'MARIA_2', 'ADAM_3', 'MARIA_3', 'ADAM_4', 'MARIA_4', 'ADAM_5', 'MARIA_5', 'ADAM_6', 'MARIA_6', 'ADAM_7', 
                  'MARIA_7', 'ADAM_8', 'MARIA_8', 'ADAM_9', 'MARIA_9', 'ADAM_10', 'MARIA_10', 'ADAM_11', 'MARIA_11', 'ENDING'],
    'partTimes': [40.505, 56.678, 56.688, 64.743, 59.355, 58.085, 56.379, 64.487, 62.353, 
                  61.563, 57.115, 59.493, 57.019, 56.283, 59.941, 57.679, 58.043, 59.707, 60.827, 59.663, 59.109, 60.443, 58.832, 64.433]
}

storyTimeDict1c = {
    'partNames': ['INTRODUCTION', 'KASIA_1', 'JANEK_1', 'KASIA_2', 'JANEK_2', 'KASIA_3', 'JANEK_3', 'KASIA_4', 'JANEK_4', 'KASIA_5', 'JANEK_5', 'KASIA_6', 'JANEK_6', 'KASIA_7', 
                  'JANEK_7', 'KASIA_8', 'JANEK_8', 'KASIA_9', 'JANEK_9', 'KASIA_10', 'JANEK_10',  'KASIA_11', 'JANEK_11', 'ENDING'],
    'partTimes': [35.257, 61.296, 59.525, 56.655, 59.92, 65.147, 57.574, 61.99, 65.329, 60.965, 
                  60.454, 57.947, 64.176, 60.955, 60.241, 59.472, 57.115, 59.387, 57.306, 58.693, 62.267, 58.063, 61.295, 55.909]
}
Question =  ["Na chwilę obecną jaki jest twój stan uwagowy?",
             "Wciśnij odpowiedni przycisk:",
                    " ",
                    "%1 - Jeśli jesteś całkowicie zaangażowany w słuchaną historię. %",
                    "%2 - Jeśli myślisz o Historii ale jej nie słuchasz %",
                    "%3 - Jeśli myślisz o czymś zupełnie innym niż Historia %",
                    "%4 - Jeśli nie jesteś w stanie prześledzić swojego toku myśli %",]

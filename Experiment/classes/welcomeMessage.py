import pygame
from pygame.locals import * 


class welcomeMessage:
    """
    Class that governs the displaying of inter-trial messages to participants (those that are not contingent on Trials )
    """

    def __init__(self, textList,font,screen,currentStage,nextStage, msgCount = 0, marginAdjust = 0,):
        self.textList  = textList
        self.msgCount = msgCount
        self.font = font
        self.screen = screen
        self.height = screen.get_size()[1]
        self.width =  screen.get_size()[0]

        self.currentStage = currentStage;
        self.nextStage = nextStage #What Kind of stage do you expect next 


    def render_multiline_text(self,text_lines, line_height=50):
        """
        Render multi-line text with dynamic left margin (in pixels) and special centering for lines with '# #'.
        """
        # Determine the width of the longest line in pixels
        max_line_pixel_width = max(self.font.size(line.replace("#", "").strip())[0] for line in text_lines)

        # Calculate the dynamic left margin as a proportion of the unused space
        margin = (self.width - max_line_pixel_width) // 2

        # Calculate the Y position to center the entire block vertically
        total_text_height = len(text_lines) * line_height
        y_offset = (self.height - total_text_height) // 2

        for i, line in enumerate(text_lines):
            # Check if the line should be centered (contains '# #')
            if line.startswith("#") and line.endswith("#"):
                clean_line = line.replace("#", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.width // 2, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)

            elif line.startswith("%") and line.endswith("%"):
                clean_line = line.replace("%", "")  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(topleft=(self.width // 4, y_offset + i * line_height))
                self.screen.blit(text_surface, text_rect)
            else:
                # Left-aligned text at the dynamic margin
                text_surface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (margin, y_offset + i * line_height))

    def run(self):
        #self.screen.fill((255/2,255/2, 255/2))

        self.render_multiline_text(self.textList[self.msgCount])
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):  # Quit
                return "exit"

            if event.type == KEYDOWN and event.key == K_SPACE:  # Progress on SPACE press
                self.msgCount += 1
                

                if self.msgCount >= len(self.textList):
                    return  self.nextStage # return flag for the next experiment


        return  self.currentStage  # return flag for welcome period


def generateMessages(firstEntityName):


    WelcomeMessage1 =  [
                        "#Welcome to the experimental procedure!#",
                        "",
                        "The purpose of this procedure is to investigate sustained attention during an auditory task.",
                        "",
                        "Participation is voluntary, and all data are anonymous and confidential.",
                        "You may withdraw from the study at any time without providing a reason.",
                        " ",
                        "#Press SPACE to continue#"]
   
    WelcomeMessage11 = [ "In a moment, you will hear a story lasting approximately 40 minutes.",
                        "While listening, please keep your gaze",
                        "fixed on the cross displayed on the screen.",
                        " ",
                        "After listening, you will be asked to verbally",
                        "recall ONLY the parts concerning selected story Character.",
                        " ",
                        "#Press SPACE to continue#"]

    WelcomeMessage2 =  ["The story will occasionally be interrupted by a question",
                        "displayed in the center of the screen:",
                        " ",
                        "#\"To what extent were you distracted from listening to the story at this moment?\"#",
                        " ",
                        "In response, use the mouse to select a point on a continuous scale",
                        "representing your subjective level of distraction from listening to the story:",
                        " ",
                        "%The beginning of the scale indicates complete attention to the story%",
                        "%The end of the scale indicates complete distraction from the story%",
                        " ",
                        "Our experiment aims to study attention, so do not hesitate to report being distracted.",
                        "Please evaluate your attentional state as honestly as possible.",
                        "",
                        "#Press SPACE to continue#"]

    WelcomeMessage2b =  ["After responding, you will be asked to verbally recall",
                        "the content of your thoughts since the last interruption of the story.",
                        " ",
                        "After pressing SPACE, the recording will begin. Additionally,",
                        "the speaking time will be displayed on the screen. Please limit your response to 25 seconds.",
                        "If you exceed the time limit, the procedure will continue automatically,",
                        "and your recording up to that point will be saved.",
                        "",
                        "The recording is completely anonymous. If your thoughts are private,",
                        "please describe them only to the extent that you feel comfortable.",
                        "",
                        "Try to report your thoughts honestly, including those unrelated to the task,",
                        "and be as detailed as possible about their course.",
                        "",
                        "#Press SPACE to continue#"]

    WelcomeMessage2c = ["We will now begin a practice session to familiarize you with the procedure.",
                        "You will hear a story unrelated to the main experiment.",
                        "",
                        "Put on your headphones and get ready.",
                        "",
                        "Press SPACE to begin the practice session"]

    WelcomeMessage3 =  [f"In a moment, you will proceed to the eye tracker calibration.",
                        "Direct your gaze toward the objects displayed on the screen.",
                        "",
                        "#Press SPACE to begin calibration#"]


    WelcomeMessage21alt =  [f"In a moment, you will hear the main story. Put on your headphones and get ready.",
                            "Please remember all parts concerning",
                            "the character named:",
                            "",
                            f"#{firstEntityName}#",
                            "",
                            "Remember that after the experiment, you will be asked to recall",
                            "details concerning only this character.",
                            " ",
                            "#Press SPACE to begin listening to the story#"]

    WelcomeMessage4 =  [f"In a moment, you will proceed to the eye tracker recalibration.",
                        "This time, the procedure will continue automatically once calibration is complete.",
                        "Direct your gaze toward the objects displayed on the screen.",
                        "",
                        "#Press SPACE to begin calibration#"]

    exitMessage1 =          ["#Thank you for participating in the study#",
                            "",
                            "You are now invited to complete a short",
                            "questionnaire"]

    
    return [WelcomeMessage1,WelcomeMessage11,WelcomeMessage2,WelcomeMessage2b,WelcomeMessage2c,WelcomeMessage3,WelcomeMessage21alt,WelcomeMessage4,exitMessage1]

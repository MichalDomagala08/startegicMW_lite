import pygame
from pygame.locals import * 


class welcomeMessage:


    def __init__(self, textList,font,screen,currentStage,nextStage, msgCount = 0, marginAdjust = 0,):
        self.textList  = textList
        self.msgCount = msgCount
        self.font = font
        self.screen = screen
        self.height = screen.get_size()[1]
        self.width =  screen.get_size()[0]

        self.currentStage = currentStage;
        self.nextStage = nextStage #What Kind of stage do you expect next 


    def render_multiline_text(self,text_lines, line_height=40):
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
                clean_line = line.replace("#", "").strip()  # Remove '# #' for rendering
                text_surface = self.font.render(clean_line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.width // 2, y_offset + i * line_height))
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


firstEntityName = "Kasia"
secondEntityName = "Jacek"


WelcomeMessage1 =  [
                             "#Witaj w procedurze Eksperymentalnej!#",
                    "",
                    "Za chwilę usłyszysz dwie historie trwające po ok 23 minuty,",
                    "które będą co jakiś czas przerywane przez pytanie, wyswietlone",
                    "na Środku ekranu:", 
                    " ",
                             "#\"Czy jesteś skupiony na zadaniu?\"#",
                    " ",
                    "Po odsłuchaniu każdej z Historii zostaniesz poproszony",
                    "o jej ustne odpamiętanie.",
                    " ",
                              "# Naciśnij SPACJĘ aby przejść dalej #"]

WelcomeMessage2 =  ["Na pytanie: \"Czy jesteś skupiony na zadaniu?\" musisz przycisnąć klawisz:.", 
                    " A (Tak) lub D (NIE). ",
                    " ",
                           "#TAK -  Jesteś całkowicie zaangażowany w słuchaną historię. #",
                           "#NIE -  Nie jesteś skupiony na na słuchanej historii.       #",
                    "Brak skupienia oznacza błądzenie myślami, brak myśli, albo", 
                    "błądzenie myślami dookoła historii ",
                    " ",
                            "# Naciśnij SPACJĘ aby przejść dalej#"]

WelcomeMessage3 =  [f"Za chwilę przejdziesz do kalibracji Okulografu.",
                     "Kieruj wzrok w stonę Obiektów na Ekranie",
                     "",
                     "# Wciśnij SPACJĘ aby rozpocząć kalibrację"]


WelcomeMessage21 =  [f"Za chwilę usłyszysz pierwszą historię.",
                     "Postaraj się skupić na wszystkie odniesienia do bohatera imieniem:",
                     "",
                                         f"#{firstEntityName}#",
                     "",
                     "# Wciśnij SPACJĘ aby zacząć słuchać historii#"]



secondStoryMessage1 =  ["Odpocznij przez kilka minut.",
                        "",
                        "Gdy będziesz gotowy aby przejsć dalej,",
                        "Naciśnij SPACJĘ"]

secondStoryMessage2 =  [f"Za chwilę przejdziesz do kalibracji Okulografu.",
                     "Kieruj wzrok w stonę Obiektów na Ekranie",
                     "",
                     "# Wciśnij SPACJĘ aby rozpocząć kalibrację"]

secondStoryMessage3 =  [f"Za chwilę usłyszysz drugą historię.",
                        "Postaraj się skupić na wszystkich odniesieniach",
                        " do bohatera imieniem:",
                        "",
                                    f"#{secondEntityName}#",
                        "",
                        "# Wciśnij SPACJĘ aby zacząć słuchać historii#"]




exitMessage1 =          ["#Dziękujemy za Udział w Badaniu#",
                        "",
                        "Zapraszamy teraz do wypełnienia krótkiego",
                        "kwestionariusza"]

import pygame
from pygame import Surface
from time import time as time_now
import sys

## Mic/Mouth
import pyaudio
import numpy
## Mouse
# import mouse

## Config
configuration = {
    'bg-color':(255,0,255),
    "width":600,
    "height":600,
    "mic-sensitivity":700
}

def main(model):
    ## Pygame
    bg = [int(x) for x in configuration['bg-color']] # green screen
    pygame.init()
    height = int(configuration["height"])
    width = int(configuration["width"])

    window = pygame.display.set_mode((width,height))#, pygame.NOFRAME)
    clock = pygame.time.Clock()
    pygame.display.set_caption('Reactive Png')

    #######
    x = 0
    while(1):
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        model()

        window.fill(bg)
        model.draw(window)

        pygame.display.update()

#######
# Bases
#######
class Callable():
    def __init__(self, fn, args=(), kwargs={}):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)

class Entity():
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Asset(Entity):
    def __init__(self,x,y,callables=[]):
        super().__init__(x, y)
        self.callables = callables

    def __call__(self) -> bool: ## True if something changed, False if not. None will be taken as True
        output = False
        for c in self.callables:
            r = c()
            output = output or r
        return output

class Frame(Asset):
    def __init__(self, image: Surface,x=0,y=0, callables=[]):
        super().__init__(x, y, callables)
        self.image = image

    def surface(self) -> Surface:
        return self.image
    def draw(self, surface : Surface, x=0, y=0):
        surface.blit(self.surface(),(x,y))

class MultiChoice(Asset):
    def __init__(self, frames: list[Frame],x=0,y=0, callables=[], i=0):
        super().__init__(x, y, callables)
        self.callables += frames
        self.frames = frames
        self.i = i
        self.default_i = i

    def surface(self) -> Surface:
        return self.frames[self.i].surface()
    def set_emotion(self, i):
        if not isinstance(i, int):
            self.i = self.default_i
            return

        self.i = i % len(self.frames)

    def draw(self, surface : Surface, x=0, y=0):
        surface.blit(self.surface(),(x,y))

#######
# Mouths
#######
class Mic():
    # Settings
    CHUNK = 1024
    WIDTH = 2
    CHANNELS = 1
    RATE = 44100

    SENSITIVITY = int(configuration['mic-sensitivity'])
    ##
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),channels=CHANNELS,rate=RATE,input=True,output=False,frames_per_buffer=CHUNK)

    def is_speaking():
        data = Mic.stream.read(Mic.CHUNK)  #read audio stream
        d = numpy.frombuffer(data, numpy.int16) # use external numpy module
        return abs(d[0]) > Mic.SENSITIVITY
    def shut():
        data = Mic.stream.read(Mic.CHUNK)  #read audio stream
        d = numpy.frombuffer(data, numpy.int16) # use external numpy module
        return abs(sum([abs(x) for x in d])) < 300

class TalkingController(Mic):
    def __init__(self,open_for=0.2,afk_for=1):
        self._mouth_delay = 0 ## Var, not const
        self._last_seen = time_now()

        self.open_for = open_for
        self.afk_for = afk_for

    def __call__(self, index, set_index_fn):
        if Mic.is_speaking():
            self._mouth_delay = time_now() + self.open_for
            self._last_seen = time_now()
            
        if self._mouth_delay < time_now():
            state = 0
        else:
            state = 1
        if self._last_seen+self.afk_for < time_now() and Mic.shut(): # Muted
            state = 2

        set_index_fn(state)
        return index != state
        
class Talker(Asset):
    def __init__(self, closed: Frame, open: Frame, muted: Frame, controller,x=0,y=0):
        super().__init__(x, y, [])
        self.frames = [closed,open,muted] ## [Closed,Open,Muted]
        self.i = 0
        self.controller = controller

    def __call__(self):
        return self.controller(self.i,self.set_frame)

    def set_frame(self,i):
        self.i = i
    def surface(self):
        return self.frames[self.i].surface()
    def draw(self,surface,dx=0,dy=0):
        surface.blit(self.surface(),(self.x + dx,self.y + dy))
from reactive_png import MultiChoice, Callable, Frame, Talker, TalkingController, main
from pygame import image
import os
import time
from camera_with_model import Camera

# class Camera:
#     def get_emotion(fn):
#         t = int(time.time()) % (21)
#         fn(t // 3)

def Vis():
    root = 'vis/'
    def load_image(path):
        return image.load(os.path.join(root,path))

    body = MultiChoice(
        frames = [
            Talker(
                *[Frame(load_image(x.lower()+y)) for y in ['.png','_speaking.png','.png']],
                controller=TalkingController()
                ) for x in [
                "Angry",
                "Disgust",
                "Fear",
                "Happy",
                "Sad",
                "Surprise",
                "Neutral"]
        ], i=6
    )
    body.callables.append(Callable(fn=Camera.get_emotion,args=(body.set_emotion,)))
    return body

main(Vis())
Video used for testing: https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Download the video, and put it into "video_input" folder, then start up the main.py file (it's all in one file)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Errors:
If you get error like:

#	img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.ANTIALIAS)
#	AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'

You need to head to ".\pythonProject\.venv\lib\site-packages\easyocr\utils.py" and switch out the "Image.ANTIALIAS" to "Image.LANCZOS" in both lines 574 and 576

If the python file does not open preview nor create usable "out_vid.py", you just have to try a couple of times, i honestly have no idea why this even happends, but im too tired to try and fix it.
If the bounding boxes of license plates(blue ones) are not on the license plates, but still follow the trajectory of their license plate, than you need to restart the program a couple times,
its a quirk of machine learning algorithm I belive.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

There are some more things that need polishing, such as showing a written text of number plate next to a bounding box, or cleaning up unneeded data.
I used pretrained models for this project, as the ones I have made were a bit too wonky, Im planning on creating them again correctly this time.

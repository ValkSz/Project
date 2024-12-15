After installing requirements start up main.py, after its done, it will create all needed information for creating bounding boxes
In file vid_create.py you can uncomment lines 59 and 60 to create a preview window, after you choose the option you can start vid_create.py file, which
will create video with bounding boxes.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Errors:
If you get error like:

#	img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.ANTIALIAS)
#	AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'

You need to head to ".\pythonProject\.venv\lib\site-packages\easyocr\utils.py" and switch out the "Image.ANTIALIAS" to "Image.LANCZOS" in both lines 574 and 576

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

There are some more things that need polishing, such as showing a written text of number plate next to a bounding box, or cleaning up unneeded data.
I used pretrained models for this project, as the ones I have made were a bit too wonky, Im planning on creating them again correctly this time.
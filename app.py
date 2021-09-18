from flask import Flask, render_template, Response, request, redirect, url_for
from camera import Video
# from train import Video1
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('login.html')

@app.route("/", methods = ['POST'])
def login():
    if request.method == "POST":
        user_name = request.form.get("name")
        password = request.form.get("pass")
        if str(user_name) == "vietdz" and str(password) == "1":
            return render_template('index.html')
        else:
            return render_template('login.html')
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#cam cho nhận diện
@app.route('/video')
def video_feed():
    return Response(gen(Video()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    # from waitress import serve
    app.run(debug=True)
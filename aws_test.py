# Name: Dong Han
# Student ID: 202111878
# Mail: dongh@mun.ca
from flask import Flask,request,render_template,jsonify
import news_summerizer
import youtube_video_summary

app = Flask(__name__)
language = 'english'

@app.route("/")
def index():
    return render_template("index_main.html")

@app.route("/index.html")
def news_summary():
    return render_template("index.html")

@app.route("/index_video.html")
def video_extract_text():
    return render_template("index_video.html")

# @app.route("/get")
# def get_bot_response():
#
#     user_text = request.args.get('msg')
#     print(user_text)
#     user_feedback = news_summerizer.summarize_for_aws(user_text)
#     return user_feedback

@app.route("/set-language", methods=['POST'])
def set_language():
    global language
    language = request.form.get('lang')

    return "Language set"

@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']
    # lange = request.form.get('lang')
    print('language:  ',language)
    bot_response = news_summerizer.summarize_for_aws(user_input,language)

    if bot_response.isdigit() and bot_response < 0:
        return jsonify({'bot_response' : -1}) if bot_response == -1 else jsonify({'bot_response' : -2})

    return jsonify({'bot_response' : bot_response})

@app.route('/processVideo', methods=['POST'])
def processVideo():
    user_input = request.form['user_input']
    bot_response = youtube_video_summary.youtube_extract_words(user_input)

    return jsonify({'bot_response' : bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

import os
os.environ['OPENAI_API_KEY']="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"
os.environ['REQUESTS_CA_BUNDLE'] = 'etc/caadmin.netskope.com.pem'

from langchain.llms import OpenAI

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)
llm = OpenAI(temperature=0.9)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   message = request.form.get('message')

   if message:
       return render_template('hello.html', message = llm(message))
   else:
       print('Request for hello page received with no message or blank message -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()

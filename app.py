from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/chat')
def handleCaht():
    return 'Chat'

if __name__ == '__main__':
    app.run()
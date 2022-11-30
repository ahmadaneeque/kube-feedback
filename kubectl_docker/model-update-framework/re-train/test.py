from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "<span style='color:red'>I am app 1</span>"


@app.route("/<name>")              # at the end point /<name>
def hello_name(name):              # call method hello_name
    return "Hello " + name

# def application(env, start_response):
#     start_response('200 OK', [('Content-Type', 'text/html')])
#     return [b"Hello World"]

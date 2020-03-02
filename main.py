from flask import Flask, render_template
from blueprints import *

app = Flask(__name__)
app.register_blueprint(home)
app.register_blueprint(upload_api)



if __name__ == '__main__':
    app.run(debug = True, host='127.0.0.1', port=5001)

import os

from blueprints import datawig
from flask import Flask

app = Flask(__name__)

app.register_blueprint(datawig.datawig)
# same secret key causes sessions to carry over from one app execution to the next
app.secret_key = os.urandom(32)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)

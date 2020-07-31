from flask import render_template, Blueprint

datawig = Blueprint('datawig', __name__, url_prefix='/datawig')


@datawig.route("/", methods=['GET'])
def index():
    return render_template(
        "datawig.html",
    )

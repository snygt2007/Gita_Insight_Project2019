from flask import Flask, request, render_template
from backend.Feature_storage import *
from flask import session
import time

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"




@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':


        file = request.files['query_img']

        # image file opened from the browser
        img = Image.open(file.stream)  # PIL image
        fake_image_dir = "./static/uploaded"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fake_file_name = 'fake_image_' + timestr +'.png'
        uploaded_img_path = os.path.join(fake_image_dir, fake_file_name)

        img.save(uploaded_img_path)

        # query = fe.extract(img)
        # dists = np.linalg.norm(features - query, axis=1)  # Do search
        # ids = np.argsort(dists)[:30] # Top 30 results
        # scores = [(dists[id], img_paths[id]) for id in ids]
        # Get the similar looking trademarks with similarity score 
        scores, similar_images = search_similar_images(fake_image_dir,fake_file_name)

        uploaded_img_path_query = '/static/uploaded/' + fake_file_name

        return render_template('index.html',
                               query_paths=uploaded_img_path_query,
                               score=scores,
                               image_paths=similar_images)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    session.clear()
    return response

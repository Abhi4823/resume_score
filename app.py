from flask import Flask, render_template, request
from utils.nlp_utils import process_resume, calculate_similarity, get_missing_keywords
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    match_score = None
    missing_keywords = []

    if request.method == 'POST':
        resume_file = request.files['resume']
        job_desc = request.form['job_desc']

        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(resume_path)

        resume_text = process_resume(resume_path)
        match_score = calculate_similarity(resume_text, job_desc)
        missing_keywords = get_missing_keywords(resume_text, job_desc)

    return render_template('index.html', score=match_score, missing=missing_keywords)

if __name__ == '__main__':
    app.run(debug=True)

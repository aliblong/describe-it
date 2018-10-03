from flask import render_template, request, Markup
import pandas as pd
import describe_it

from web import app


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/output')
def output():
    subject = request.args.get('subject')
    top_features_and_descriptors = \
        describe_it.top_features_and_descriptors(subject)
    output = []
    for feature, descriptors in top_features_and_descriptors.items():
        output.append('<br>')
        output.append(f'{feature.capitalize()}:')
        for descriptor, mult in descriptors:
            output.append(f'<div class="entrytext"><p>{descriptor} ({mult})</p></div>')
        output.append('</br>')
    html_output = Markup('\n'.join(output))

    return render_template('output.html', output=html_output)

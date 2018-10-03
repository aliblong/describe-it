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
    for i, (feature, descriptors) in enumerate(
            top_features_and_descriptors.items()
            ):
        feature_descriptions = []
        #color = '#aaa' if i % 2 == 0 else '#bbb'
        #feature_descriptions.append(f'<div class="col-md-3 feature" style="background-color:{color};"><h4>{feature.capitalize()}:</h4><div class="descriptor">')
        feature_descriptions.append(f'<div class="col-md-3"><h4>{feature.capitalize()}:</h4><div class="descriptor">')
        for descriptor, mult in descriptors:
            feature_descriptions.append(
                    f'<p>{descriptor} ({mult})</p>'
            )
        feature_descriptions.append(f'</div></div>')
        output.append(''.join(feature_descriptions))
    html_output = Markup(''.join(output))

    return render_template(
            'output.html',
            subject=subject,
            output=html_output
    )

from flask import Flask, render_template, session, redirect, url_for

from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

from wtforms import IntegerField, StringField, SubmitField, SelectField, DecimalField
from wtforms.validators import Required


# Initialize Flask App
app = Flask(__name__)


print("loading my model")
# with open('model.pkl', 'rb') as handle:
#     machine_learning_model = pickle.load(handle)
print("model loaded")

'creeping_blade_pressure'
'dry_end_temperature'
'jet_speed'
'load_machine_chest_refiner'
'machine_chest_consistency'
'main_steam_pressure'
'main_steam_temperature'
'silo_level'
'stock_pump_flow'
'wet_end_temperature'
'wire_speed'
'yankee_inlet_steam_pressure'



# Initialize Form Class
class theForm(Form):
    param1 = DecimalField(label='creeping_blade_pressure:', places=2, validators=[Required()])
    param2 = DecimalField(label='jet_speed:', places=2, validators=[Required()])
    param3 = DecimalField(label='load_machine_chest_refiner:', places=2, validators=[Required()])
    param4 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param5 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param6 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param7 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param8 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param9 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param10 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param11 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    param12 = DecimalField(label='machine_chest_consistency:', places=2, validators=[Required()])
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def home():
    print(session)
    form = theForm(csrf_enabled=False)
    if form.validate_on_submit():  # activates this if when i hit submit!
        # Retrieve values from form
        session['sepal_length'] = form.param1.data
        session['sepal_width'] = form.param2.data
        session['petal_length'] = form.param3.data
        session['petal_width'] = form.param4.data
        # Create array from values
        flower_instance = [(session['sepal_length']), (session['sepal_width']), (session['petal_length']),
                           (session['petal_width'])]

        # Return only the Predicted iris species
        flowers = ['setosa', 'versicolor', 'virginica']
        # session['prediction'] = flowers[machine_learning_model.predict(flower_instance)[0]]

        # Implement Post/Redirect/Get Pattern
        return redirect(url_for('home'))

    return render_template('home.html', form=form, **session)


# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

app.secret_key = 'super_secret_key_shhhhhh'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
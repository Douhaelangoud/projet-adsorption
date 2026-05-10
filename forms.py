from flask_wtf import FlaskForm, RecaptchaField
from wtforms import PasswordField, StringField, TextAreaField
from wtforms.validators import DataRequired, Email, Length

class SignupForm(FlaskForm):
     username = StringField('Username', validators=[DataRequired()])
     firstname = StringField('First Name', validators=[DataRequired()])
     email = StringField('Email', validators=[DataRequired(), Email()])
     password = PasswordField('Password', validators=[DataRequired()])

     recaptcha = RecaptchaField()






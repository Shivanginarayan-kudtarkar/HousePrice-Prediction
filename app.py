import pandas as pd
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, flash
import pickle
import uuid
import mysql.connector
from werkzeug.security import generate_password_hash  # (not used, but ok)
from hashlib import sha256

# -------------------------------------------------
# Flask app setup
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "Shivani123"

# -------------------------------------------------
# Load House Price model
# -------------------------------------------------
with open("house_price_regression_dataset.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------
# MySQL connection config  (XAMPP)
# -------------------------------------------------
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",              # default XAMPP password
    "database": "house_price_db" # your DB name
    # add "port": 3307 here if your MySQL runs on 3307 instead of 3306
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def hash_password(password: str) -> str:
    """Hash password using sha256 (same style as IPL code)."""
    return sha256(password.encode()).hexdigest()

# -------------------------------------------------
# Prediction function for House Price
# -------------------------------------------------
def predict_score(
    Square_Footage=1360,
    Num_Bedrooms=2,
    Num_Bathrooms=1,
    Year_Built=1981,
    Lot_Size=0.599637,
    Garage_Size=1,
    Neighborhood_Quality=5
):
    temp_array = [Square_Footage, Num_Bedrooms, Num_Bathrooms,
                  Year_Built, Lot_Size, Garage_Size, Neighborhood_Quality]

    temp_array = np.array([temp_array])
    print("Model input:", temp_array)

    # model.predict returns an array → take first value and convert to int
    return int(model.predict(temp_array)[0])

# -------------------------------------------------
# Routes
# -------------------------------------------------

# Home – show index page (only if logged in)
@app.route('/')
def index():
    return render_template('index.html')



# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # 🚨 User NOT logged in → redirect to login
    if 'user_id' not in session:
        flash("Please login to use the prediction tool", "error")
        return redirect(url_for("login"))

    # 🚨 If logged in → allow prediction
    if request.method == 'POST':
        Square_Footage = float(request.form['Square_Footage'])
        Num_Bedrooms = int(request.form['Num_Bedrooms'])
        Num_Bathrooms = int(request.form['Num_Bathrooms'])
        Year_Built = int(request.form['Year_Built'])
        Lot_Size = float(request.form['Lot_Size'])
        Garage_Size = int(request.form['Garage_Size'])
        Neighborhood_Quality = int(request.form['Neighborhood_Quality'])

        prediction = predict_score(
            Square_Footage,
            Num_Bedrooms,
            Num_Bathrooms,
            Year_Built,
            Lot_Size,
            Garage_Size,
            Neighborhood_Quality
        )

        return render_template("result.html", predicted_price=prediction)

    return render_template("prediction.html")



# Registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # check if email already exists
        cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            flash("Email address already registered.", "error")
            return redirect(url_for("register"))

        # insert new user
        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO user (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed),
        )
        conn.commit()
        cursor.close()
        conn.close()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    # GET
    return render_template("registration.html")  # make sure this template exists


# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and hash_password(password) == user["password"]:
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            session["uemail"] = user["email"]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))

@app.route('/about')
def about():
    return render_template('about.html')



# -------------------------------------------------
# Run app
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7001)

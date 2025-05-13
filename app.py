from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

df = pd.read_csv("C:/Users/ASUS/Documents/Project/processor_pediction.csv")  # Update with your dataset path
df = df.drop(['Unnamed: 4', 'Unnamed: 5'], axis=1)
df = df.dropna()
KNN = RandomForestClassifier()
x = df[['Age', 'Task', 'Profession']]
y = df['Processor_name']
KNN = KNN.fit(x, y)

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_processor = None
    
    if request.method == 'POST':
        # Get user input from the form
        age = int(request.form['age'])
        task = int(request.form['task'])
        profession = int(request.form['profession'])

        # Perform prediction using your model
        test_data = pd.DataFrame({'Age': [age], 'Task': [task], 'Profession': [profession]})
        predicted_processor = KNN.predict(test_data)[0]

    return render_template('app.html', predicted_processor=predicted_processor)

if __name__ == '__main__':
    app.run(debug=True)

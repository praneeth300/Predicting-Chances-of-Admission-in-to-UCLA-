from flask import Flask,render_template,request
import pickle

#intialize the app


app=Flask(__name__)

@app.route('/',methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            gre_score= int(request.form['gre_score'])
            toefl_score= int(request.form['toefl_score'])
            university_rating= int(request.form['university_rating'])
            sop= float(request.form['sop'])
            lor= float(request.form['lor'])
            cgpa= float(request.form['cgpa'])
            is_research=request.form['research']
            if (is_research=='yes'):
                research=1
            else:
                research=0
            filename='Linear_model.pickle'
            regression=pickle.load(open(filename,'rb'))
            prediction=regression.predict([[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]])
            print('prediction is',prediction)

            return render_template("results.html",prediction=round(100*prediction[0]))
        except Exception as e:
            print('The Exception is:',e)
            return 'Something is Wrong'

    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

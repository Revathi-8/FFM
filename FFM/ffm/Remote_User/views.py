from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,predict_flood_forecasting,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_flood_forecasting_Detection(request):
    if request.method == "POST":

        if request.method == "POST":

            UEI= request.POST.get('UEI')
            Start_Date= request.POST.get('Start_Date')
            End_Date= request.POST.get('End_Date')
            Duration= request.POST.get('Duration')
            Location= request.POST.get('Location')
            Districts= request.POST.get('Districts')
            State= request.POST.get('State')
            Latitude= request.POST.get('Latitude')
            Longitude= request.POST.get('Longitude')
            Severity= request.POST.get('Severity')
            Area_Affected= request.POST.get('Area_Affected')
            Human_fatality= request.POST.get('Human_fatality')
            Human_injured= request.POST.get('Human_injured')
            Human_Displaced= request.POST.get('Human_Displaced')
            Animal_Fatality= request.POST.get('Animal_Fatality')
            Description_of_Casualties_injured= request.POST.get('Description_of_Casualties_injured')
            Extent_of_damage= request.POST.get('Extent_of_damage')
            Event_Source= request.POST.get('Event_Source')
            Event_Souce_ID= request.POST.get('Event_Souce_ID')

        df = pd.read_csv('Datasets.csv', encoding='latin-1')

        def apply_response(label):
            if (label == 'Heavy Rain'):
                return 0  # Heavy rains
            elif (label == 'Cyclone and Flood'):
                return 1  # Cyclone and Flood

        df['Label'] = df['Main_Cause'].apply(apply_response)

        cv = CountVectorizer()
        X = df['UEI'].apply(str)
        y = df['Label']

        print("UEI")
        print(X)
        print("Label")
        print(y)

        X = cv.fit_transform(X)

        #X = cv.fit_transform(df['CHASSIS_NO'].apply(lambda x: np.str_(X)))

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print("ACCURACY")
        print(naivebayes)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_nb))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Gradient Boosting Classifier")

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            X_train,
            y_train)
        clfpredict = clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, clfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, clfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, clfpredict))
        models.append(('GradientBoostingClassifier', clf))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        UEI1 = [UEI]
        vector1 = cv.transform(UEI1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'Heavy Rain'
        elif (prediction == 1):
            val = 'Cyclone and Flood'

        print(val)
        print(pred1)

        predict_flood_forecasting.objects.create(
        UEI=UEI,
        Start_Date=Start_Date,
        End_Date=End_Date,
        Duration=Duration,
        Location=Location,
        Districts=Districts,
        State=State,
        Latitude=Latitude,
        Longitude=Longitude,
        Severity=Severity,
        Area_Affected=Area_Affected,
        Human_fatality=Human_fatality,
        Human_injured=Human_injured,
        Human_Displaced=Human_Displaced,
        Animal_Fatality=Animal_Fatality,
        Description_of_Casualties_injured=Description_of_Casualties_injured,
        Extent_of_damage=Extent_of_damage,
        Event_Source=Event_Source,
        Event_Souce_ID=Event_Souce_ID,
        Prediction=val)

        return render(request, 'RUser/Predict_flood_forecasting_Detection.html',{'objs': val})
    return render(request, 'RUser/Predict_flood_forecasting_Detection.html')




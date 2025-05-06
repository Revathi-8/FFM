from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class predict_flood_forecasting(models.Model):

    UEI= models.CharField(max_length=3000)
    Start_Date= models.CharField(max_length=3000)
    End_Date= models.CharField(max_length=3000)
    Duration= models.CharField(max_length=3000)
    Location= models.CharField(max_length=3000)
    Districts= models.CharField(max_length=3000)
    State= models.CharField(max_length=3000)
    Latitude= models.CharField(max_length=3000)
    Longitude= models.CharField(max_length=3000)
    Severity= models.CharField(max_length=3000)
    Area_Affected= models.CharField(max_length=3000)
    Human_fatality= models.CharField(max_length=3000)
    Human_injured= models.CharField(max_length=3000)
    Human_Displaced= models.CharField(max_length=3000)
    Animal_Fatality= models.CharField(max_length=3000)
    Description_of_Casualties_injured= models.CharField(max_length=3000)
    Extent_of_damage= models.CharField(max_length=3000)
    Event_Source= models.CharField(max_length=3000)
    Event_Souce_ID= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)




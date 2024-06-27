from django.db import models
from djongo import models



class Appliances(models.Model):
    modelNumber = models.CharField(max_length=100,primary_key=True)
    name = models.CharField(max_length=100)
    year = models.IntegerField()
    maker = models.CharField(max_length=100)
    price = models.IntegerField()
    
    def __str__(self):
        return f"Appliances: {self.modelNumber} - {self.name} - {self.year} - {self.maker} - {self.price}"



    def appliances_same_maker_homes():
        
        queryset = any
        return queryset
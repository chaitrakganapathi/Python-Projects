from django.db import models
from djongo import models


class Owners(models.Model):
    SSN = models.CharField(max_length=100, primary_key=True)
    Name = models.CharField(max_length=100)
    Dependents = models.IntegerField()
    Income = models.IntegerField()
    Age = models.IntegerField()
    Profession = models.CharField(max_length=100)
    
    def __str__(self):
        return f"SSN: {self.SSN}, Name: {self.Name}, Dependents: {self.Dependents}, Income: {self.Income}, Age: {self.Age}, Profession: {self.Profession}"










from django.db import models
from djongo import models
#from webapp.owners import Owners
from webapp.appliances import Appliances
from django.db.models import Max , Count, Q
from djongo.models import ArrayReferenceField, DjongoManager

class Home_appliances(models.Model):
    modelNumber = models.CharField(max_length=100,primary_key=True,db_column='modelNumber')
    
    def __str__(self):
        return f"Home_appliances: {self.modelNumber}"
    


class Homes(models.Model):
    homeId = models.IntegerField(db_column='HomeId', primary_key=True)
    floorSpace = models.IntegerField(db_column='FloorSpace')
    floors = models.IntegerField(db_column='Floors')
    bedRooms = models.IntegerField(db_column='BedRooms')
    bathRooms = models.IntegerField(db_column='BathRooms')
    landSize = models.IntegerField(db_column='LandSize')
    yearConstructed = models.IntegerField(db_column='YearConstructed')
    
    htype = models.CharField(max_length=100, db_column='Type')
    price = models.IntegerField(db_column='Price')
    #owner = models.CharField(max_length=100, )
    owner = models.ForeignKey("Owners", on_delete=models.CASCADE,db_column='Owner')
    #owner = models.ForeignKey(Owners, on_delete=models.CASCADE)
    city = models.CharField(max_length=100, db_column='City')
    preowned = models.IntegerField(db_column='Preowned')
    #appliances= models.CharField(max_length=100,db_column='Appliances')
    #Appliances = models.ArrayField(model_container=models.CharField(max_length=100),blank=True)
    #appliances = models.ForeignKey("Appliances",on_delete=models.CASCADE,db_column='Appliances')
    appliances = ArrayReferenceField(
        to=Appliances,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        db_column='Appliances'
    )
    
    
    agentID=models.IntegerField(db_column='agentID')
    status=models.CharField(max_length=100)
    

    def __str__(self):
        return f"Home ID: {self.homeId}, Type: {self.htype}, Price: {self.price}, Owner: {self.owner}, City: {self.city}, Appliances: {self.appliances}"

 
    @staticmethod
    def save_new_home(floorSpace, floors, bedRooms, bathRooms, landSize, yearConstructed, homeId, htype, price, owner, city, preowned, appliances,status):
    # Create a new Homes object with the desired values
     new_home = Homes(
        floorSpace=floorSpace,
        floors=floors,
        bedRooms=bedRooms,
        bathRooms=bathRooms,
        landSize=landSize,
        yearConstructed=yearConstructed,
        homeId=homeId,
        htype=htype,
        price=price,
        owner=owner,
        city=city,
        preowned=preowned,
        appliances=appliances,
        status=status,
    )
        # Save the new_home object to the database
     saved_home= new_home.save()
     return saved_home
    
    
    @staticmethod
    def find_homes_all_eppliance_same_maker(maker):
        appliances_with_same_maker = Appliances.objects.filter(maker=maker)
        appliance_ids = [app.modelNumber for app in appliances_with_same_maker]
        print("appliances id",appliance_ids)
        homes=[]
        homes_with_appliances = Homes.objects.filter(appliances__in=appliance_ids)
        #homes_with_appliances = Homes.objects.filter(appliances__in=appliance_ids)
        for home in homes_with_appliances:
            home_appliance_ids = [app.modelNumber for app in home.appliances.all()]
            if set(home_appliance_ids).issubset(set(appliance_ids)):
                home.appliances=home_appliance_ids
                homes.append(home)
                print("appls",home.appliances)
        print("query",homes_with_appliances.query)
        print("appliances_with_same_maker",appliances_with_same_maker)
        
            
        return homes

    @staticmethod
    def get_homes_sold_more_than_once():
        queryset=Homes.objects.filter(preowned__gt=1)
        #To print the query string
        print(str(queryset.query))
        return queryset

    
    @staticmethod
    def get_most_expensive_home_for_owner():
        # queryset = Homes.objects.filter(owner=owner_name)
        # queryset=queryset.order_by('-price')
        queryset=Homes.objects.order_by('-price')
        most_expensive_home = queryset.first()
        print(queryset.query.__str__())  # Print the generated SQL query
        return queryset.filter(price=most_expensive_home.price)
    @staticmethod
    def get_homes_owned_by_owner_in_city(owner_name, city_name):
        queryset = Homes.objects.filter(owner=owner_name, city=city_name)
        return queryset
    
    @staticmethod
    def get_homes_filtered(owner, city, price, home_t):
        filtered_homes = Homes.objects.all()
        if owner:
            filtered_homes = filtered_homes.filter(owner=owner)

        if city:
            filtered_homes = filtered_homes.filter(city=city)

        if home_t:
            filtered_homes = filtered_homes.filter(htype=home_t )

        if price:
            filtered_homes = filtered_homes.filter(price__lte=price)

        # queryset = Homes.objects.filter(owner=owner).filter(city=city).filter(htype=home_t).filter(price__lt=price)
        # print(queryset)
        return filtered_homes
    
    @staticmethod
    
    def get_People_With_Apts_Mansions():
        apartments = set(Homes.objects.filter(htype="Apartment").values_list("owner", flat=True).distinct())
        mansions = set(Homes.objects.filter(htype="Mansion").values_list("owner", flat=True).distinct())
        print(type(apartments))
        print(mansions)
        # Find the intersection of owners who own both types
        users_with_both_types = apartments.intersection(mansions)
        # Retrieve homes owned by users with both types'
        homes = Homes.objects.filter(owner__in=users_with_both_types).filter(htype__in=['Apartment', 'Mansion'])

        return homes
    
    @staticmethod
    def get_owners_of_most_expensive_homes_in_city():
        # Step 1: Group homes by city and find the maximum price for each city.
        max_prices_per_city = Homes.objects.values('city').annotate(max_price=Max('price'))
        print(max_prices_per_city)
        #, the annotate() function is used to add aggregate values to each object in a queryset.
        # Step 2: Loop through each city and retrieve the homes with the maximum price.
        expensive_homes= []
        for city_info in max_prices_per_city:
            city = city_info['city']
            max_price = city_info['max_price']
            city_expensive_homes = Homes.objects.filter(city=city, price=max_price).distinct()
            expensive_homes.extend(list(city_expensive_homes))
           #extend is used to append the list

        #return owners_of_expensive_homes_by_city
        return expensive_homes
    
    @staticmethod
    def get_homes_below_price_in_given_city(city, price):
        # city='Troy'
        # price = 222
        queryset = Homes.objects.filter(city=city, price__lte=int(price))
        return queryset
    
    @staticmethod
    def find_homes_for_sale(city, min_bedrooms=1, min_bathrooms=1, max_price=10000000, hometype='Mansion'):
        print("Inside model")
        queryset = Homes.objects.filter(city=city , preowned=1)  # Assuming preowned=1 indicates homes for sale
        
        print("queryset",queryset.query)
        
        for home in queryset:
            print("Home ID1:", home.id)
            print("City:", home.city)
            print("Bedrooms:", home.bedRooms)
            print("Bathrooms:", home.bathRooms)
            print("Price:", home.price)
            print("Home Type:", home.htype)

        if min_bedrooms:
            queryset = queryset.filter(bedRooms__gte=int(min_bedrooms))

        if min_bathrooms:
            queryset = queryset.filter(bathRooms__gte=int(min_bathrooms))

        if max_price:
            queryset = queryset.filter(price__lte=int(max_price))
            
        if hometype:
            queryset = queryset.filter(htype=hometype)

        for home in queryset:
            print("Home ID:", home.id)
            print("City:", home.city)
            print("Bedrooms:", home.bedRooms)
            print("Bathrooms:", home.bathRooms)
            print("Price:", home.price)
            print("Home Type:", home.htype)
        return queryset

   
    
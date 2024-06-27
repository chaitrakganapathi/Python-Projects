import pymongo


from django.shortcuts import render
from webapp.owners import Owners
from webapp.homes import Homes
from webapp.agents import Agents
from webapp.transactions import Transactions
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
from django.db.models import Max , Count, Q
from django.shortcuts import render , get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
from django.http import HttpResponse , HttpResponseNotFound



def home(request):
    context={}
    return render(request, "webapp/index.html")

#def home(request):
    context={}
    
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["CSI5450"]
    collection = db["webapp_homes"]

    # Retrieve data from MongoDB
    #chdata_from_mongo1 = collection.find({}, {"_id": 0, "owner": 1}) # Assuming you want to retrieve the "name" field
    # data_from_mongo = collection.distinct('owner') # Assuming you want to retrieve the "name" field
    # print(data_from_mongo)
    #chdata_from_mongo = set(item['owner'] for item in data_from_mongo1)
    


    return render(request, "webapp/index.html", {'data_from_mongo': data_from_mongo})
    
@csrf_exempt    
def addAgent(request):
    context={}
    if request.method == 'POST':
        # Retrieve field values from the form data
        aid = request.POST.get('AgentID')
        name = request.POST.get('Name')
        cRate = request.POST.get('CommissionRate')
        company = request.POST.get('Company')
    Agents.save_new_agent(aid,name,cRate,company)
    return render(request, "webapp/index.html",context)

@csrf_exempt
def addHome(request):
    context={}
    if request.method == 'POST':
         # Retrieve field values from the form data
         flrSpace = request.POST.get('floorSpace')
         flrs = request.POST.get('floors')
         bdRooms = request.POST.get('bedRooms')
         bthRooms = request.POST.get('bathRooms')
         landSize = request.POST.get('landSize')
         yearConstd = request.POST.get('homeYear')
         hid = request.POST.get('homeId')
         htyp = request.POST.get('homeType')
         price = request.POST.get('homePrice')
         #owner = request.POST.get('owner')
         owner = request.POST.get('owner')  # Retrieve owner's name
         city = request.POST.get('city')
         preowned = 0
         appliances = request.POST.get('appliances')
         status='Available'
    Homes.save_new_home(flrSpace,flrs,bdRooms,bthRooms,landSize,yearConstd,hid,htyp,price,owner,city,preowned,appliances,status)
         
    return render(request, "webapp/index.html",context)

@csrf_exempt
def move_from_sale_to_owned(request):
    context={}
    if request.method == 'POST':
        homeId = request.POST.get('homeId')
        new_owner_id = request.POST.get('owner')
        new_price = request.POST.get('new_price')

    #Retrieve the home object
        home = get_object_or_404(Homes, homeId=homeId)
        old_owner= home.owner.SSN
        
        print("old",old_owner)
        #Check if the home is currently available for sale
        if home.status == 'Available':
            try:
                
                # Retrieve the new owner object
                new_owner = Owners.objects.get(SSN=new_owner_id)
                
                Transactions.save_transaction(home.homeId,home.agentID,old_owner,old_owner,new_owner_id,new_price,'Owned',home.htype)

                #Update home status to 'owned'
                home.status = 'Owned'

                #Assign the new owner to the home
                home.owner = new_owner
                home.price=new_price
                home.preowned=home.preowned+1

                 #Save changes to the home object
                out=home.save()
                print("output ",out)
                

                return HttpResponse("Home moved to owned list successfully.")
            except Owners.DoesNotExist:
                return HttpResponse("New owner does not exist", status=404)
        else:
            return HttpResponse("Home is not available for sale", status=400)

    return HttpResponse("Invalid request", status=400)


@csrf_exempt
def make_home_owner(request):
    context={}
    if request.method == 'POST':
        #hid = request.POST.get('homeId')
        #owner = request.POST.get('owner')
        homeId = request.POST.get('homeId')
        new_price = request.POST.get('new_price')
        new_owner_id = request.POST.get('owner')
        try:
            new_owner = Owners.objects.get(SSN=new_owner_id)
        except Owners.DoesNotExist:
            ownername = request.POST.get('ownername')
            owner_age = request.POST.get('owner_age')
            owner_profession = request.POST.get('owner_profession')
            if ownername:
                          new_owner = Owners.objects.create(
                SSN=new_owner_id,
                Name=ownername,
                Age=owner_age,
                Profession=owner_profession
            )
            else:
                return HttpResponse("Owner's name is required", status=400)

        try:
            # Fetch the Home instance to update
            #homeId = Homes.objects.get(homeId=homeId)
            home = get_object_or_404(Homes,homeId=homeId)
            old_owner=home.owner.SSN
            # Update the owner of the home
            home.owner = new_owner
            home.price=new_price
            home.preowned=home.preowned+1
            home.status = "owned" # Assuming you want to mark the home as owned if it wasn't already
            home.save()

            # Return a success response
            Transactions.save_transaction(home.homeId,home.agentID,old_owner,old_owner,new_owner_id,new_price,'Owned',home.htype)
            return HttpResponse("Home owner updated successfully.")

        except Homes.DoesNotExist:
            return HttpResponse("Home not found", status=404)

        except Owners.DoesNotExist:
            return HttpResponse("Owner not found", status=404)

    else:
        # Return an error response if the request method is not POST
        return HttpResponse("Method not allowed", status=405)
       


#q10- Find homes that up for sale in a given city that meet certain buyer choices such as number of bedrooms, baths, etc
@csrf_exempt
def filter_homes(request):
    context={}
    print("Inside views")
    if request.method == 'POST':
        bdRooms = request.POST.get('bedrooms')
        bthRooms = request.POST.get('bathrooms')
        htyp = request.POST.get('home-type')
        price = request.POST.get('max-price')
        city = request.POST.get('city')
        result=Homes.find_homes_for_sale(city,bdRooms,bthRooms,price,htyp)
        if result is None or len(result) == 0:
            result = 'No homes found matching the criteria.'
        else:
            context['homes'] = result
    print("After model")
    return render(request, "webapp/jsondisplaypage.html",{'documents': result})

#q5- Find owners who do not own the homes they used to own. 
@csrf_exempt
def find_owners_q5(request):
    context={}
    query_result=Transactions.find_owners_who_dont_own_previous_homes()
    print("query result", query_result)
    context['dict']=query_result
    return render(request, "webapp/jsondisplaypage.html",{'dict': query_result})

#q1- List all the homes owned by a given owner in a given city. 
@csrf_exempt
def list_homes_owned_by_owner_in_city(request):
    context = {}
    if request.method == 'POST':
        owner_name = request.POST.get('owner')
        city_name = request.POST.get('city')
        homes = Homes.get_homes_owned_by_owner_in_city(owner_name, city_name)
        context['homes'] = homes
    return render(request, "webapp/jsondisplaypage.html", {'messages': homes})

#q2-List all the homes that were sold more than once
@csrf_exempt
def homes_sold_morethan_one_q2(request):
    context = {}
    if request.method == 'POST':
        homes = Homes.get_homes_sold_more_than_once()
        context['homes'] = homes
    return render(request, "webapp/jsondisplaypage.html", {'documents': homes})

#q3- Find the most expensive home an owner ever bought. appliance_maker_q4
@csrf_exempt
def expensive_home_q3(request):
    context = {}
    if request.method == 'POST':
        query_result = Homes.get_most_expensive_home_for_owner()
        context['homes'] = query_result
    return render(request, "webapp/jsondisplaypage.html", {'messages': query_result})

#q4- Find all the homes that include all e appliances by the same maker.
@csrf_exempt
def appliance_maker_q4(request):
    context = {}
    if request.method == 'POST':
        maker= request.POST.get('maker')
        query_result = Homes.find_homes_all_eppliance_same_maker(maker)
        print("result",query_result)
        context['homes'] = query_result
    return render(request, "webapp/jsondisplaypage.html", {'documents': query_result})

#q6- Find the total commissions earned by an agent. Assume that commission earned is on the purchased price of a home he/she sells. 
@csrf_exempt
def agent_commission_q6(request):
    context = {}
    if request.method == 'POST':
        id = request.POST.get('agentId')
        query_result = Agents.get_commission(id)
        msg = "Total Commission earned is "+str(query_result)
    return render(request, "webapp/jsondisplaypage.html", {'result': msg})

#q7- Find people who own apartments as well as mansions.
@csrf_exempt
def apartment_and_mansion_q7(request):
    context = {}
    if request.method == 'POST':
        query_result = Homes.get_People_With_Apts_Mansions()
        context['homes'] = query_result
    return render(request, "webapp/jsondisplaypage.html", {'messages': query_result})

#q8- List all the homes below a price in a given city.expensive_home_every_city_q9
@csrf_exempt
def home_lt_price_q8(request):
    context = {}
    if request.method == 'POST':
        price = request.POST.get('price')
        city_name = request.POST.get('city')
        homes = Homes.get_homes_below_price_in_given_city(city_name,price)
        context['homes'] = homes
    return render(request, "webapp/jsondisplaypage.html", {'messages': homes})

#q9- List owners who own all the most expensive homes in a given city
@csrf_exempt
def expensive_home_every_city_q9(request):
    context = {}
    if request.method == 'POST':
         city = request.POST.get('city')
         query_result = Homes.get_owners_of_most_expensive_homes_in_city()
         context['homes'] = query_result
    return render(request, "webapp/jsondisplaypage.html", {'owners': query_result})

@csrf_exempt
def search_homes(request):
    city = request.GET.get('city')
    min_bedrooms = request.GET.get('min_bedrooms')
    min_bathrooms = request.GET.get('min_bathrooms')
    max_price = request.GET.get('max_price')

    homes = Homes.find_homes_for_sale(city, min_bedrooms, min_bathrooms, max_price,'Apartment')
    
    context = {
        'homes': homes
    }
    return render(request, 'home_search_results.html', context) 


@csrf_exempt    
def searchQuery(request):
    context = {}
    agent=None
    home_t=None
    # if request.method == 'POST':
    if request.POST.get('city-select')!=None:
        city=request.POST.get('city-select')
    if request.POST.get('price-select')!=None:
        price=request.POST.get('price-select')
    if request.POST.get('owner-select')!=None:
        owner=request.POST.get('owner-select')
    if request.POST.get('agent-select')!=None:
        agent=request.POST.get('agent-select')
    if request.POST.get('home-type-select')!=None:
        home_t=request.POST.get('home-type-select')
    
    # city='Troy'
    # price = 222
    print(city)
    query_result = Homes.get_homes_filtered(owner, city, price, home_t)
    
    return render(request, "webapp/jsondisplaypage.html", {'documents': query_result})

      
@csrf_exempt    
def preDefinedQuery(request):
    context={}

    if request.method == 'POST':
        selected_option = request.POST.get('query')

        if selected_option == 'q1':
            query_result=Homes.get_homes_two_floors(2)

        elif selected_option == 'q2':
            query_result=Homes.get_homes_two_floors(2)
            
        elif selected_option == 'q3':
            query_result=Homes.get_homes_two_floors(2)
    
        elif selected_option == 'q4':
            query_result=Homes.get_People_With_Apts_Mansions()
            homes_data = []
            for home in query_result:
                home_info = {
                    'id': home.homeId,
                    'type': home.htype
                }
                homes_data.append(home_info)
            print(homes_data)
            # query_result = homes_data
            # homes_data = [{'id': home.homeId, 'type': home.htype} for home in query_result1]
            # query_result=JsonResponse({"homes":homes_data})

            
        elif selected_option == 'q5':
            #query_result=Homes.get_homes_two_floors(2)
            query_result=Agents.display_agent()
            print("Result from two floors1",query_result)
            agent_data = []
            for agent in query_result:
                agent_info = {
                    'id': agent.AgentID,
                    'name': agent.Name,
                    'commission rate':agent.CommissionRate
                }
                agent_data.append(agent_info)
            print(agent_data)
            
        elif selected_option == 'q6':
            query_result=Homes.get_homes_two_floors(2)
        
        elif selected_option == 'q7':
            query_result = Homes.get_most_expensive_home_for_owner()
        
        elif selected_option == 'q8':
            city=request.POST.get('city-select')
            price=request.POST.get('price-select')
            query_result = Homes.get_homes_below_price_in_given_city(city, price)
           
        else: 
            query_result=Homes.get_homes_two_floors(2)
            print("Result from two floors2")
            for home in query_result:
                print(home)
    
    return render(request, "webapp/jsondisplaypage.html", {'documents': query_result})
    
    
        

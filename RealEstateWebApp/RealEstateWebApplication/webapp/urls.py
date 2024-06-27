from . import views
from django.urls import path

urlpatterns = [
    path("",views.home, name="home"),
    path("filter_homes",views.filter_homes, name="filter_homes"),
    path("find_owners_q5",views.find_owners_q5, name="find_owners_q5"),
    path("agent_commission_q6",views.agent_commission_q6, name="agent_commission_q6"),
    path("list_homes_owned_by_owner_in_city",views.list_homes_owned_by_owner_in_city, name="list_homes_owned_by_owner_in_city"),
    path("expensive_home_q3",views.expensive_home_q3, name="expensive_home_q3"),
    path("appliance_maker_q4",views.appliance_maker_q4, name="appliance_maker_q4"),
    path("homes_sold_morethan_one_q2",views.homes_sold_morethan_one_q2, name="homes_sold_morethan_one_q2"),
    path("home_lt_price_q8",views.home_lt_price_q8, name="home_lt_price_q8"),
    path("expensive_home_every_city_q9",views.expensive_home_every_city_q9, name="expensive_home_every_city_q9"),
    path("apartment_and_mansion_q7",views.apartment_and_mansion_q7, name="apartment_and_mansion_q7"),
    path("twoFloors",views.preDefinedQuery, name="preDefinedQuery"),
    #path("preDefinedQuery",views.preDefinedQuery, name="preDefinedQuery"),
    path("searchQuery",views.searchQuery, name="searchQuery"),
    path("addAgent",views.addAgent, name="addAgent"),
    path("addHome",views.addHome, name="addHome"),
    path("move_from_sale_to_owned",views.move_from_sale_to_owned, name="move_from_sale_to_owned"),
    path("make_home_owner",views.make_home_owner, name="make_home_owner"),
    
    
]



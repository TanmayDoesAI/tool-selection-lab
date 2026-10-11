"""Every run owns a deep copy; there is no clock, network, or global mutable state."""
from copy import deepcopy

BASE = {
 'orders': {'O100': {'customer_id':'C1','status':'processing','items':[{'sku':'MUG','quantity':2}], 'total':24.0,'shipping_address':'14 Oak Road','note':'','priority':'normal'}, 'O101':{'customer_id':'C2','status':'shipped','items':[{'sku':'BAG','quantity':1}], 'total':40.0,'shipping_address':'9 Elm Road','note':'','priority':'normal'}},
 'inventory': {'MUG':{'on_hand':20,'reserved':4,'reorder_point':5,'warehouse':'east'},'BAG':{'on_hand':3,'reserved':1,'reorder_point':4,'warehouse':'west'}},
 'shipments': {'S1':{'order_id':'O101','status':'in_transit','carrier':'ParcelCo','tracking':'PK123','eta':'2026-10-09','address':'9 Elm Road','weight_kg':1.2,'service':'standard','delivered_at':None}},
 'customers': {'C1':{'name':'Alex Lee','email':'alex.one@example.test','address':'14 Oak Road','consent':False,'tier':'silver','points':120,'credit':0.0,'tags':[]},'C2':{'name':'Alex Lee','email':'alex.two@example.test','address':'9 Elm Road','consent':True,'tier':'gold','points':400,'credit':5.0,'tags':['vip']}},
 'invoices': {'I1':{'order_id':'O100','amount':24.0,'paid':24.0,'refunded':0.0,'status':'paid','due_date':'2026-10-12','memo':''},'I2':{'order_id':'O101','amount':40.0,'paid':0.0,'refunded':0.0,'status':'open','due_date':'2026-10-10','memo':''}},
 'tickets': {'T1':{'customer_id':'C1','order_id':'O100','subject':'Damaged mug','status':'open','priority':'normal','assignee':'unassigned','tags':[],'messages':[],'sla_hours':24},'T2':{'customer_id':'C2','order_id':'O101','subject':'Where is my bag?','status':'open','priority':'high','assignee':'Maya','tags':[],'messages':[],'sla_hours':8}},
 'campaigns': {'K1':{'name':'Autumn','status':'draft','budget':100.0,'spend':20.0,'impressions':1000,'clicks':100,'conversions':5,'segment':'all','coupon':'FALL10'}},
 'coupons': {'FALL10':{'percent':10,'active':True,'uses':2}},
 'products': {'MUG':{'name':'Ceramic Mug','price':12.0,'cost':5.0,'active':True,'description':'Blue ceramic mug','weight_kg':0.4,'tags':['kitchen'],'category':'home','supplier':'V1'},'BAG':{'name':'Canvas Bag','price':40.0,'cost':18.0,'active':True,'description':'Reusable canvas bag','weight_kg':1.2,'tags':['travel'],'category':'accessories','supplier':'V2'}},
 'returns': {'R1':{'order_id':'O101','sku':'BAG','quantity':1,'reason':'wrong size','status':'requested','received':False,'inspection':'pending','label':None,'fee':0.0}},
 'events':[], 'notes':[], 'receipts':[], 'reservations':[]
}

def fresh(): return deepcopy(BASE)

def record(state, table, key):
    if key not in state[table]: raise ValueError(f'Unknown {table} identifier: {key}')
    return state[table][key]

def update(state, table, key, **changes):
    row = record(state,table,key)
    before = deepcopy(row)
    row.update(changes)
    state['events'].append({'table':table,'id':key,'before':before,'changes':changes})
    return deepcopy(row)

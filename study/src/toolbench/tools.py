"""100 distinct business contracts, with shared validation/storage primitives."""
from dataclasses import dataclass
from copy import deepcopy
import json
from typing import Callable
from jsonschema import Draft202012Validator
from .world import record as r, update as u

@dataclass
class Tool:
    name: str
    category: str
    description: str
    parameters: dict
    behavior: Callable
    example: dict
    request: str
    mutates: bool = False
    @property
    def schema(self):
        return {'type':'function','function':{'name':self.name,'description':self.description,'parameters':self.parameters}}
    def execute(self,state,args):
        Draft202012Validator(self.parameters).validate(args)
        return deepcopy(self.behavior(state,**args))

TOOLS = {}
CATEGORIES = {
 'orders':'Order records, line items, cancellation, delivery address, fulfillment holds and priorities.',
 'inventory':'Warehouse quantities, available versus reserved stock, replenishment, allocation and stock receipts.',
 'shipping':'Parcel tracking, delivery forecasts, shipping labels, carriers, addresses and transport rates.',
 'customers':'Customer identity, profile, contact information, loyalty points, store credit and consent.',
 'billing':'Invoices, payment collection, financial refunds, balances, due dates and payment reminders.',
 'support':'Helpdesk tickets, replies, assignments, escalation, SLA and customer issue histories.',
 'marketing':'Campaigns, audience segments, coupons, budget, engagement and attribution.',
 'catalog':'Product descriptions, prices, costs, classifications, weights, suppliers and active listings.',
 'returns':'Physical return authorization, labels, warehouse receipt, inspection, fees and exchanges.',
 'analytics':'Computed retail sales, margins, payment exposure, inventory health and operational summaries.'}

def add(cat,name,desc,fields,example,request,fn,mutates=False):
    props={}
    for key, kind in fields.items():
        if isinstance(kind,list): props[key]={'type':'string','enum':kind}
        else:
            props[key]={'type':kind,'description':key.replace('_',' ')}
            if kind in ('integer','number'): props[key]['minimum']=0
            if kind=='string': props[key]['minLength']=1
    TOOLS[name]=Tool(name,cat,desc,{'type':'object','properties':props,'required':list(fields),'additionalProperties':False},fn,example,request,mutates)

S='string'; N='number'; I='integer'; B='boolean'
def rows(s,t,**filters): return [{'id':k,**v} for k,v in s[t].items() if all(v.get(f)==x for f,x in filters.items())]
def field(s,t,key,*names): return {f:r(s,t,key)[f] for f in names}
def append(s,t,data):
    s[t].append(data); return deepcopy(data)
def new(s,t,prefix,data):
    key=f'{prefix}{len(s[t])+1}'; s[t][key]=data; s['events'].append({'table':t,'id':key,'created':True}); return {'id':key,**data}

# Orders: reads, status changes, and line-item edits are distinct contracts.
add('orders','get_order','Read a full sales order by order ID.',{'order_id':S},{'order_id':'O100'},'Show the order record for O100.',lambda s,order_id:r(s,'orders',order_id))
add('orders','list_customer_orders','Find all orders belonging to a customer.',{'customer_id':S},{'customer_id':'C1'},'Which purchases belong to C1?',lambda s,customer_id:rows(s,'orders',customer_id=customer_id))
add('orders','get_order_total','Read the billed sales total, not an outstanding invoice balance.',{'order_id':S},{'order_id':'O100'},'What was the selling total of O100?',lambda s,order_id:field(s,'orders',order_id,'total'))
add('orders','get_order_items','Read ordered SKUs and quantities.',{'order_id':S},{'order_id':'O101'},'Which items and quantities are in O101?',lambda s,order_id:field(s,'orders',order_id,'items'))
def cancel(s,order_id):
    if r(s,'orders',order_id)['status']=='shipped': raise ValueError('Shipped orders cannot be cancelled; use returns')
    return u(s,'orders',order_id,status='cancelled')
add('orders','cancel_order','Cancel an unshipped order; does not issue a financial refund.',{'order_id':S},{'order_id':'O100'},'Cancel the unshipped purchase O100.',cancel,True)
add('orders','hold_order','Pause fulfillment by placing an order on hold.',{'order_id':S},{'order_id':'O100'},'Put fulfillment of O100 on hold.',lambda s,order_id:u(s,'orders',order_id,status='held'),True)
add('orders','release_order_hold','Resume fulfillment of a held order.',{'order_id':S},{'order_id':'O100'},'Set O100 back to processing.',lambda s,order_id:u(s,'orders',order_id,status='processing'),True)
add('orders','set_order_address','Change delivery address on an unshipped sales order, not a parcel.',{'order_id':S,'address':S},{'order_id':'O100','address':'21 Pine Road'},'Send O100 to 21 Pine Road.',lambda s,order_id,address:u(s,'orders',order_id,shipping_address=address) if r(s,'orders',order_id)['status']!='shipped' else (_ for _ in ()).throw(ValueError('Already shipped')),True)
add('orders','set_order_priority','Set warehouse fulfillment priority.',{'order_id':S,'priority':['normal','urgent']},{'order_id':'O100','priority':'urgent'},'Mark O100 urgent for the warehouse.',lambda s,order_id,priority:u(s,'orders',order_id,priority=priority),True)
add('orders','add_order_note','Attach an internal sales-order note.',{'order_id':S,'note':S},{'order_id':'O100','note':'Gift wrap'},'Attach Gift wrap to O100 as an internal note.',lambda s,order_id,note:u(s,'orders',order_id,note=note),True)

add('inventory','get_stock','Read physical and reserved warehouse stock.',{'sku':S},{'sku':'MUG'},'Show warehouse counts for MUG.',lambda s,sku:r(s,'inventory',sku))
add('inventory','get_available_stock','Compute sellable quantity = on-hand minus reserved.',{'sku':S},{'sku':'MUG'},'How many MUG units are free to promise after allocations?',lambda s,sku:{'available':r(s,'inventory',sku)['on_hand']-r(s,'inventory',sku)['reserved']})
add('inventory','list_low_stock','List SKUs below their reorder point using available stock.',{}, {},'Which products need replenishment?',lambda s: [{'sku':k,**v} for k,v in s['inventory'].items() if v['on_hand']-v['reserved']<v['reorder_point']])
def reserve(s,sku,quantity):
    row=r(s,'inventory',sku)
    if quantity>row['on_hand']-row['reserved']: raise ValueError('Insufficient available stock')
    return u(s,'inventory',sku,reserved=row['reserved']+quantity)
add('inventory','reserve_stock','Allocate available stock without changing physical count.',{'sku':S,'quantity':I},{'sku':'MUG','quantity':3},'Allocate three MUG units for future fulfillment.',reserve,True)
def release(s,sku,quantity):
    row=r(s,'inventory',sku)
    if quantity>row['reserved']: raise ValueError('Cannot release more than reserved')
    return u(s,'inventory',sku,reserved=row['reserved']-quantity)
add('inventory','release_stock','Release reserved warehouse units to sellable stock.',{'sku':S,'quantity':I},{'sku':'MUG','quantity':2},'Free two allocated MUG units.',release,True)
add('inventory','receive_stock','Add a physical supplier receipt to on-hand units.',{'sku':S,'quantity':I},{'sku':'BAG','quantity':5},'Receive five new BAG units from the supplier.',lambda s,sku,quantity:u(s,'inventory',sku,on_hand=r(s,'inventory',sku)['on_hand']+quantity),True)
add('inventory','set_reorder_point','Change the replenishment threshold.',{'sku':S,'quantity':I},{'sku':'BAG','quantity':8},'Replenish BAG when available stock drops below eight.',lambda s,sku,quantity:u(s,'inventory',sku,reorder_point=quantity),True)
add('inventory','transfer_stock','Move the SKU stock location to another warehouse.',{'sku':S,'warehouse':['east','west']},{'sku':'MUG','warehouse':'west'},'Move MUG stock to the west warehouse.',lambda s,sku,warehouse:u(s,'inventory',sku,warehouse=warehouse),True)
add('inventory','get_stock_value','Compute warehouse value at product cost, not retail price.',{'sku':S},{'sku':'MUG'},'What is the cost value of physical MUG stock?',lambda s,sku:{'value':r(s,'inventory',sku)['on_hand']*r(s,'products',sku)['cost']})
add('inventory','audit_stock','Compare a counted physical quantity with the recorded quantity.',{'sku':S,'counted':I},{'sku':'BAG','counted':4},'Compare four counted BAG units with the ledger.',lambda s,sku,counted:{'recorded':r(s,'inventory',sku)['on_hand'],'counted':counted,'difference':counted-r(s,'inventory',sku)['on_hand']})

add('shipping','get_shipment','Read a parcel record by shipment ID.',{'shipment_id':S},{'shipment_id':'S1'},'Show parcel S1.',lambda s,shipment_id:r(s,'shipments',shipment_id))
add('shipping','find_order_shipments','Find parcels attached to a sales order.',{'order_id':S},{'order_id':'O101'},'Find parcels for O101.',lambda s,order_id:rows(s,'shipments',order_id=order_id))
add('shipping','track_parcel','Read delivery status and carrier tracking number.',{'shipment_id':S},{'shipment_id':'S1'},'Where is S1 and what tracking code should I use?',lambda s,shipment_id:field(s,'shipments',shipment_id,'status','tracking'))
add('shipping','estimate_delivery','Read the promised arrival date.',{'shipment_id':S},{'shipment_id':'S1'},'When should S1 arrive?',lambda s,shipment_id:field(s,'shipments',shipment_id,'eta'))
add('shipping','quote_shipping','Calculate domestic transport fee from parcel weight and service.',{'weight_kg':N,'service':['standard','express']},{'weight_kg':2,'service':'express'},'Quote express transport for a two kilogram parcel.',lambda s,weight_kg,service:{'fee':round(4+weight_kg*(5 if service=='express' else 2),2)})
add('shipping','set_parcel_address','Redirect an existing parcel; does not change order billing address.',{'shipment_id':S,'address':S},{'shipment_id':'S1','address':'21 Pine Road'},'Redirect parcel S1 to 21 Pine Road.',lambda s,shipment_id,address:u(s,'shipments',shipment_id,address=address),True)
add('shipping','set_carrier','Assign a parcel to a transport provider.',{'shipment_id':S,'carrier':['ParcelCo','QuickShip']},{'shipment_id':'S1','carrier':'QuickShip'},'Give S1 to QuickShip.',lambda s,shipment_id,carrier:u(s,'shipments',shipment_id,carrier=carrier),True)
add('shipping','mark_delivered','Record delivery confirmation with an ISO date.',{'shipment_id':S,'date':S},{'shipment_id':'S1','date':'2026-10-09'},'Confirm S1 was delivered on 2026-10-09.',lambda s,shipment_id,date:u(s,'shipments',shipment_id,status='delivered',delivered_at=date),True)
add('shipping','create_shipping_label','Create an outbound parcel from an unshipped order.',{'order_id':S,'service':['standard','express']},{'order_id':'O100','service':'standard'},'Create standard outbound shipping for O100.',lambda s,order_id,service:new(s,'shipments','S',{'order_id':order_id,'status':'label_created','carrier':'ParcelCo','tracking':f'PK{len(s["shipments"])+1}','eta':'2026-10-10','address':r(s,'orders',order_id)['shipping_address'],'weight_kg':sum(r(s,'products',i['sku'])['weight_kg']*i['quantity'] for i in r(s,'orders',order_id)['items']),'service':service,'delivered_at':None}),True)
add('shipping','list_delivery_exceptions','List parcels whose transport status is exception.',{}, {},'Show parcels with delivery exceptions.',lambda s:rows(s,'shipments',status='exception'))

add('customers','get_customer','Read customer profile by stable customer ID.',{'customer_id':S},{'customer_id':'C1'},'Show C1 profile.',lambda s,customer_id:r(s,'customers',customer_id))
add('customers','search_customers','Search names case-insensitively; names may identify multiple people.',{'name':S},{'name':'Alex Lee'},'Find customers called Alex Lee.',lambda s,name:[{'id':k,**v} for k,v in s['customers'].items() if name.lower() in v['name'].lower()])
add('customers','get_customer_by_email','Resolve exact email to a customer profile.',{'email':S},{'email':'alex.two@example.test'},'Who has email alex.two@example.test?',lambda s,email:rows(s,'customers',email=email))
add('customers','set_customer_email','Replace a customer contact email.',{'customer_id':S,'email':S},{'customer_id':'C1','email':'new@example.test'},'Set C1 email to new@example.test.',lambda s,customer_id,email:u(s,'customers',customer_id,email=email),True)
add('customers','set_customer_address','Change customer default address, not an existing order or parcel.',{'customer_id':S,'address':S},{'customer_id':'C1','address':'21 Pine Road'},'Set the default address for C1 to 21 Pine Road.',lambda s,customer_id,address:u(s,'customers',customer_id,address=address),True)
add('customers','set_marketing_consent','Set customer opt-in permission for marketing.',{'customer_id':S,'consent':B},{'customer_id':'C1','consent':True},'Opt C1 into marketing.',lambda s,customer_id,consent:u(s,'customers',customer_id,consent=consent),True)
add('customers','get_loyalty_balance','Read loyalty tier and point balance, not cash credit.',{'customer_id':S},{'customer_id':'C1'},'How many reward points does C1 have, and what tier?',lambda s,customer_id:field(s,'customers',customer_id,'points','tier'))
add('customers','award_loyalty_points','Add earned loyalty points.',{'customer_id':S,'points':I},{'customer_id':'C1','points':30},'Award C1 thirty reward points.',lambda s,customer_id,points:u(s,'customers',customer_id,points=r(s,'customers',customer_id)['points']+points),True)
add('customers','issue_store_credit','Add store credit in dollars, without refunding a payment.',{'customer_id':S,'amount':N},{'customer_id':'C1','amount':7},'Give C1 seven dollars of store credit.',lambda s,customer_id,amount:u(s,'customers',customer_id,credit=r(s,'customers',customer_id)['credit']+amount),True)
add('customers','tag_customer','Add a segmentation label to a customer.',{'customer_id':S,'tag':S},{'customer_id':'C1','tag':'wholesale'},'Label C1 wholesale.',lambda s,customer_id,tag:u(s,'customers',customer_id,tags=sorted(set(r(s,'customers',customer_id)['tags']+[tag]))),True)

add('billing','get_invoice','Read invoice, payment and refunded amounts.',{'invoice_id':S},{'invoice_id':'I1'},'Show invoice I1.',lambda s,invoice_id:r(s,'invoices',invoice_id))
add('billing','find_order_invoices','Find billing invoices linked to a sales order.',{'order_id':S},{'order_id':'O100'},'Which invoice belongs to O100?',lambda s,order_id:rows(s,'invoices',order_id=order_id))
add('billing','get_invoice_balance','Compute remaining amount payable.',{'invoice_id':S},{'invoice_id':'I2'},'What remains to be paid on I2?',lambda s,invoice_id:{'balance':r(s,'invoices',invoice_id)['amount']-r(s,'invoices',invoice_id)['paid']})
def pay(s,invoice_id,amount):
    row=r(s,'invoices',invoice_id)
    if amount>row['amount']-row['paid']: raise ValueError('Payment exceeds balance')
    paid=row['paid']+amount
    return u(s,'invoices',invoice_id,paid=paid,status='paid' if paid==row['amount'] else 'open')
add('billing','record_payment','Apply a local payment to an outstanding invoice.',{'invoice_id':S,'amount':N},{'invoice_id':'I2','amount':40},'Record forty dollars paid on I2.',pay,True)
def refund(s,invoice_id,amount):
    row=r(s,'invoices',invoice_id)
    if amount>row['paid']-row['refunded']: raise ValueError('Refund exceeds unrefunded payments')
    return u(s,'invoices',invoice_id,refunded=row['refunded']+amount)
add('billing','refund_payment','Refund a captured payment, not authorize a physical return or store credit.',{'invoice_id':S,'amount':N},{'invoice_id':'I1','amount':12},'Refund twelve dollars of the payment on I1; no physical return.',refund,True)
add('billing','set_invoice_due_date','Change invoice payment deadline.',{'invoice_id':S,'date':S},{'invoice_id':'I2','date':'2026-10-20'},'Extend I2 deadline to 2026-10-20.',lambda s,invoice_id,date:u(s,'invoices',invoice_id,due_date=date),True)
add('billing','add_invoice_memo','Attach an accounting memo.',{'invoice_id':S,'memo':S},{'invoice_id':'I1','memo':'Reviewed'},'Put Reviewed on I1 as an accounting memo.',lambda s,invoice_id,memo:u(s,'invoices',invoice_id,memo=memo),True)
add('billing','list_unpaid_invoices','List invoices with positive outstanding amount.',{}, {},'Show all bills with money still due.',lambda s:[{'id':k,**v} for k,v in s['invoices'].items() if v['amount']>v['paid']])
add('billing','send_payment_reminder','Record a fictional payment reminder in local notes.',{'invoice_id':S},{'invoice_id':'I2'},'Record a payment reminder for I2.',lambda s,invoice_id:append(s,'notes',{'kind':'payment_reminder','invoice_id':invoice_id,'balance':r(s,'invoices',invoice_id)['amount']-r(s,'invoices',invoice_id)['paid']}),True)
add('billing','get_refund_capacity','Compute maximum additional refund available.',{'invoice_id':S},{'invoice_id':'I1'},'How much more can be refunded from I1?',lambda s,invoice_id:{'refundable':r(s,'invoices',invoice_id)['paid']-r(s,'invoices',invoice_id)['refunded']})

add('support','get_ticket','Read a helpdesk case.',{'ticket_id':S},{'ticket_id':'T1'},'Show helpdesk case T1.',lambda s,ticket_id:r(s,'tickets',ticket_id))
add('support','list_customer_tickets','Find cases for one customer.',{'customer_id':S},{'customer_id':'C1'},'Which support cases belong to C1?',lambda s,customer_id:rows(s,'tickets',customer_id=customer_id))
add('support','reply_ticket','Append a local support response; never sends external mail.',{'ticket_id':S,'message':S},{'ticket_id':'T1','message':'We are investigating'},'Reply We are investigating on T1.',lambda s,ticket_id,message:u(s,'tickets',ticket_id,messages=r(s,'tickets',ticket_id)['messages']+[message]),True)
add('support','assign_ticket','Assign the helpdesk owner.',{'ticket_id':S,'agent':S},{'ticket_id':'T1','agent':'Maya'},'Assign T1 to Maya.',lambda s,ticket_id,agent:u(s,'tickets',ticket_id,assignee=agent),True)
add('support','close_ticket','Resolve a helpdesk case, without changing order or refund state.',{'ticket_id':S},{'ticket_id':'T1'},'Resolve support case T1.',lambda s,ticket_id:u(s,'tickets',ticket_id,status='closed'),True)
add('support','reopen_ticket','Return a case to the open queue.',{'ticket_id':S},{'ticket_id':'T1'},'Set case T1 to open.',lambda s,ticket_id:u(s,'tickets',ticket_id,status='open'),True)
add('support','escalate_ticket','Raise urgency to high and assign the escalation queue.',{'ticket_id':S},{'ticket_id':'T1'},'Escalate T1 to the escalation queue.',lambda s,ticket_id:u(s,'tickets',ticket_id,priority='high',assignee='escalation'),True)
add('support','tag_ticket','Attach a classification to a support case.',{'ticket_id':S,'tag':S},{'ticket_id':'T1','tag':'damage'},'Classify T1 with damage tag.',lambda s,ticket_id,tag:u(s,'tickets',ticket_id,tags=sorted(set(r(s,'tickets',ticket_id)['tags']+[tag]))),True)
add('support','get_ticket_sla','Read response deadline in hours.',{'ticket_id':S},{'ticket_id':'T2'},'What is the response window for T2 in hours?',lambda s,ticket_id:field(s,'tickets',ticket_id,'sla_hours'))
add('support','list_open_tickets','List unresolved helpdesk cases.',{}, {},'Show the unresolved case queue.',lambda s:rows(s,'tickets',status='open'))

add('marketing','get_campaign','Read campaign budget and engagement record.',{'campaign_id':S},{'campaign_id':'K1'},'Show campaign K1.',lambda s,campaign_id:r(s,'campaigns',campaign_id))
add('marketing','set_campaign_budget','Set campaign spending ceiling.',{'campaign_id':S,'amount':N},{'campaign_id':'K1','amount':200},'Set K1 spending ceiling to two hundred dollars.',lambda s,campaign_id,amount:u(s,'campaigns',campaign_id,budget=amount),True)
add('marketing','pause_campaign','Pause promotional campaign delivery.',{'campaign_id':S},{'campaign_id':'K1'},'Pause K1 advertising.',lambda s,campaign_id:u(s,'campaigns',campaign_id,status='paused'),True)
add('marketing','activate_campaign','Make a campaign active.',{'campaign_id':S},{'campaign_id':'K1'},'Activate K1 advertising.',lambda s,campaign_id:u(s,'campaigns',campaign_id,status='active'),True)
add('marketing','set_campaign_segment','Select campaign audience by customer tier.',{'campaign_id':S,'tier':['all','silver','gold']},{'campaign_id':'K1','tier':'gold'},'Target K1 only at gold tier customers.',lambda s,campaign_id,tier:u(s,'campaigns',campaign_id,segment=tier),True)
add('marketing','get_campaign_ctr','Compute clicks divided by impressions.',{'campaign_id':S},{'campaign_id':'K1'},'What share of K1 impressions became clicks?',lambda s,campaign_id:{'ctr':r(s,'campaigns',campaign_id)['clicks']/max(1,r(s,'campaigns',campaign_id)['impressions'])})
add('marketing','get_campaign_conversion_rate','Compute conversions per click, not click-through rate.',{'campaign_id':S},{'campaign_id':'K1'},'What share of K1 clicks converted?',lambda s,campaign_id:{'conversion_rate':r(s,'campaigns',campaign_id)['conversions']/max(1,r(s,'campaigns',campaign_id)['clicks'])})
add('marketing','get_coupon','Read coupon discount and redemption record.',{'code':S},{'code':'FALL10'},'Show the FALL10 promotion.',lambda s,code:r(s,'coupons',code))
add('marketing','disable_coupon','Deactivate an existing promotional code.',{'code':S},{'code':'FALL10'},'Turn off FALL10.',lambda s,code:u(s,'coupons',code,active=False),True)
add('marketing','list_opted_in_audience','List only customers consenting to marketing.',{}, {},'Who may receive marketing under their recorded consent?',lambda s:rows(s,'customers',consent=True))

add('catalog','get_product','Read product catalog record, not warehouse counts.',{'sku':S},{'sku':'MUG'},'Show the catalog listing for MUG.',lambda s,sku:r(s,'products',sku))
add('catalog','search_products','Search product names by substring.',{'query':S},{'query':'canvas'},'Find listings with canvas in their name.',lambda s,query:[{'sku':k,**v} for k,v in s['products'].items() if query.lower() in v['name'].lower()])
add('catalog','set_product_price','Change retail selling price; existing orders retain historical totals.',{'sku':S,'price':N},{'sku':'MUG','price':14},'Make MUG retail at fourteen dollars.',lambda s,sku,price:u(s,'products',sku,price=price),True)
add('catalog','set_product_cost','Change supplier unit cost used by margin calculations.',{'sku':S,'cost':N},{'sku':'MUG','cost':6},'Set the unit acquisition cost of MUG to six dollars.',lambda s,sku,cost:u(s,'products',sku,cost=cost),True)
add('catalog','set_product_description','Replace customer-facing product copy.',{'sku':S,'description':S},{'sku':'MUG','description':'Ocean blue mug'},'Use Ocean blue mug as the description of MUG.',lambda s,sku,description:u(s,'products',sku,description=description),True)
add('catalog','set_product_category','Change merchandising classification.',{'sku':S,'category':S},{'sku':'BAG','category':'travel'},'Classify BAG in travel.',lambda s,sku,category:u(s,'products',sku,category=category),True)
add('catalog','set_product_weight','Set parcel-planning unit weight in kilograms.',{'sku':S,'weight_kg':N},{'sku':'MUG','weight_kg':0.5},'Set MUG unit weight to half a kilogram.',lambda s,sku,weight_kg:u(s,'products',sku,weight_kg=weight_kg),True)
add('catalog','deactivate_product','Remove a listing from active sale; keeps warehouse stock.',{'sku':S},{'sku':'BAG'},'Stop selling BAG while retaining its inventory.',lambda s,sku:u(s,'products',sku,active=False),True)
add('catalog','tag_product','Attach a merchandising keyword.',{'sku':S,'tag':S},{'sku':'MUG','tag':'gift'},'Label MUG gift in the catalog.',lambda s,sku,tag:u(s,'products',sku,tags=sorted(set(r(s,'products',sku)['tags']+[tag]))),True)
add('catalog','get_product_margin','Compute unit retail price less supplier cost.',{'sku':S},{'sku':'MUG'},'How many dollars of unit margin does MUG earn?',lambda s,sku:{'unit_margin':r(s,'products',sku)['price']-r(s,'products',sku)['cost']})

add('returns','get_return','Read a physical merchandise return record.',{'return_id':S},{'return_id':'R1'},'Show physical return R1.',lambda s,return_id:r(s,'returns',return_id))
add('returns','list_order_returns','Find merchandise returns for an order.',{'order_id':S},{'order_id':'O101'},'Which physical returns exist for O101?',lambda s,order_id:rows(s,'returns',order_id=order_id))
add('returns','authorize_return','Approve a requested physical return; does not refund money.',{'return_id':S},{'return_id':'R1'},'Approve the physical return R1 without refunding.',lambda s,return_id:u(s,'returns',return_id,status='authorized'),True)
add('returns','reject_return','Reject a physical return request.',{'return_id':S},{'return_id':'R1'},'Reject physical return R1.',lambda s,return_id:u(s,'returns',return_id,status='rejected'),True)
add('returns','create_return_label','Generate an inbound return label, not outbound shipment.',{'return_id':S},{'return_id':'R1'},'Make an inbound mailing label for R1.',lambda s,return_id:u(s,'returns',return_id,label='RET-'+return_id),True)
add('returns','receive_return','Record that returned merchandise reached the warehouse.',{'return_id':S},{'return_id':'R1'},'Record physical receipt of R1.',lambda s,return_id:u(s,'returns',return_id,received=True,status='received'),True)
add('returns','inspect_return','Record condition of received merchandise.',{'return_id':S,'condition':['resellable','damaged']},{'return_id':'R1','condition':'damaged'},'Mark R1 inspection as damaged.',lambda s,return_id,condition:u(s,'returns',return_id,inspection=condition),True)
add('returns','set_return_fee','Set a restocking fee without issuing any payment.',{'return_id':S,'amount':N},{'return_id':'R1','amount':3},'Set R1 restocking fee to three dollars.',lambda s,return_id,amount:u(s,'returns',return_id,fee=amount),True)
def restock(s,return_id):
    row=r(s,'returns',return_id)
    if not row['received'] or row['inspection']!='resellable' or row['status']=='restocked': raise ValueError('Return must be received, resellable and not already restocked')
    u(s,'inventory',row['sku'],on_hand=r(s,'inventory',row['sku'])['on_hand']+row['quantity'])
    return u(s,'returns',return_id,status='restocked')
add('returns','restock_return','Add inspected resellable received goods to inventory once.',{'return_id':S},{'return_id':'R1'},'Restock R1 after it has been received and inspected as resellable.',restock,True)
add('returns','get_return_refund_quote','Calculate retail return value less restocking fee; does not issue refund.',{'return_id':S},{'return_id':'R1'},'Estimate net retail value to refund for R1.',lambda s,return_id:{'refund_quote':max(0,r(s,'products',r(s,'returns',return_id)['sku'])['price']*r(s,'returns',return_id)['quantity']-r(s,'returns',return_id)['fee'])})

add('analytics','sales_summary','Aggregate noncancelled historical sales orders.',{}, {},'What are aggregate historical sales and order count?',lambda s:{'revenue':sum(v['total'] for v in s['orders'].values() if v['status']!='cancelled'),'orders':sum(v['status']!='cancelled' for v in s['orders'].values())})
add('analytics','sales_by_customer','Aggregate historical sales for one customer.',{'customer_id':S},{'customer_id':'C1'},'How much has C1 purchased in total?',lambda s,customer_id:{'revenue':sum(v['total'] for v in s['orders'].values() if v['customer_id']==customer_id and v['status']!='cancelled')})
add('analytics','sales_by_sku','Aggregate units sold for a SKU from order lines.',{'sku':S},{'sku':'MUG'},'How many MUG units were sold historically?',lambda s,sku:{'units':sum(i['quantity'] for v in s['orders'].values() if v['status']!='cancelled' for i in v['items'] if i['sku']==sku)})
add('analytics','inventory_summary','Aggregate physical, reserved and available units across SKUs.',{}, {},'Summarize physical, reserved and available warehouse units.',lambda s:{'on_hand':sum(v['on_hand'] for v in s['inventory'].values()),'reserved':sum(v['reserved'] for v in s['inventory'].values()),'available':sum(v['on_hand']-v['reserved'] for v in s['inventory'].values())})
add('analytics','receivables_summary','Aggregate outstanding invoice exposure.',{}, {},'What total invoice exposure remains unpaid?',lambda s:{'outstanding':sum(v['amount']-v['paid'] for v in s['invoices'].values())})
add('analytics','refund_summary','Aggregate refunded captured payments.',{}, {},'How many dollars have been refunded overall?',lambda s:{'refunded':sum(v['refunded'] for v in s['invoices'].values())})
add('analytics','support_summary','Count open cases by urgency.',{}, {},'Summarize open support cases and high urgency cases.',lambda s:{'open':sum(v['status']=='open' for v in s['tickets'].values()),'high':sum(v['status']=='open' and v['priority']=='high' for v in s['tickets'].values())})
add('analytics','shipping_summary','Count total and delivered parcels.',{}, {},'How many parcels exist and how many arrived?',lambda s:{'total':len(s['shipments']),'delivered':sum(v['status']=='delivered' for v in s['shipments'].values())})
add('analytics','marketing_summary','Aggregate promotion spend, clicks and conversions.',{}, {},'Summarize advertising spend, clicks and conversions.',lambda s:{k:sum(v[k] for v in s['campaigns'].values()) for k in ('spend','clicks','conversions')})
add('analytics','customer_summary','Count customers, consent and total loyalty points.',{}, {},'Summarize customer count, opted-in count and reward points.',lambda s:{'customers':len(s['customers']),'opted_in':sum(v['consent'] for v in s['customers'].values()),'points':sum(v['points'] for v in s['customers'].values())})
assert len(TOOLS)==100

#coding=utf8
import sys
import math
import hashlib

infile=sys.argv[1]
outfile=sys.argv[2]

fin=open(infile, 'r')
lines=fin.readlines()
fout=open(outfile, 'w')

features = ['lasthour_poi_agg_poi_id_coor_count', 'area_id', '10min', 'bill_number_per_rider', 'poi_agg_poi_id_mean_food_total_value', 'lasthour_poi_agg__avg_speed_meanpoi_id', 'poi_agg_poi_id_std_dd', 'poi_lng_bin', 'poi_lat_bin', 'poi_agg_poi_id_mean_arriveshop_cost', 'lasthour_poi_agg_poi_id_mean_arrive_guest_cost', 'customer_latitude', '_10min_not_fetched_order_num', 'area_id_area_id#hour#minute_arriveshop_cost_std', '_10min_notbusy_working_rider_num', 'lasthour_poi_agg_poi_id_mean_box_total_value', 'poi_agg_poi_id_mean_box_total_value', 'lasthour_poi_agg_poi_id_arrive_guest_avg_speed_mean', 'poi_agg_poi_id_mean_waiting_order_num', 'delivery_distance', '_10min_working_rider_num', 'poi_agg_poi_id_arrive_guest_avg_speed_mean', 'area_id_area_id#hour#minute_food_total_value_sum', 'lasthour_poi_agg__avg_speed_stdpoi_id', 'cst_lng_bin', 'poi_agg_poi_id_std_fetch_cost', 'lasthour_poi_agg_poi_id_mean_waiting_order_num', 'area_id_area_id#hour#minute_arriveshop_cost_mean', 'lasthour_poi_agg_poi_id_mean_dd', 'poi_agg_poi_id_arrive_guest_avg_speed_std', 'lasthour_poi_agg_poi_id_mean_food_total_value', 'cst_lat_bin', 'poi_agg_poi_id_std_arrive_guest_cost', 'lasthour_poi_agg_poi_id_std_arriveshop_cost', 'waiting_order_num', '_10min_deliverying_order_num', 'direction', 'poi_lng', 'lasthour_poi_agg_poi_id_mean_fetch_cost', 'lasthour_poi_agg_poi_id_std_dd', 'poi_agg_poi_id_mean_arrive_guest_cost', 'customer_longitude', 'poi_agg_poi_id_coor_count', 'poi_lat', 'poi_agg_poi_id_mean_fetch_cost', 'poi_agg__avg_speed_meanpoi_id', 'food_num', 'poi_agg__avg_speed_stdpoi_id', 'poi_id', 'lasthour_poi_agg_poi_id_std_fetch_cost', 'lasthour_poi_agg_poi_id_mean_arriveshop_cost', 'lasthour_poi_agg_poi_id_std_arrive_guest_cost', 'food_total_value', 'prd_arrive_guest_time', 'poi_agg_poi_id_std_arriveshop_cost', 'lasthour_poi_agg_poi_id_arrive_guest_avg_speed_std', 'box_total_value', 'poi_agg_poi_id_mean_dd', 'weekday', '15min', 'lasthour_poi_agg_poi_id_mean_delivery_distance', 'area_busy_coef', 'poi_agg_poi_id_mean_delivery_distance']
fea_index = [0 for i in features]
label = 'delivery_duration'
lab_index = -1

def round_to_n(x, n):
    if x==0.0:
        return 0.0
    return round(x, -long(math.floor(math.log10(abs(x))))+n-1) 
def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1
def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field, hashstr(feat, nr_bins)) for (field, feat) in feats]
    #feats = ['{0}:{1}:1'.format(field, feat) for (field, feat) in feats]
    return feats

for i,line in enumerate(lines):
    line=line.strip()
    if i==0:
        fea_names=line.split(",")
        for j,fea in enumerate(fea_names):
            if fea in features:
                fea_index[features.index(fea)]=j
            if fea == label:
                lab_index = j
        assert len(fea_index) == len(features)
        print zip(features, fea_index)
        if lab_index == -1:#test_sub文件没有label
            lab_index = fea_names.index('order_id')
    else:
        li=line.split(",")
        feats=[]
        for k,fea_name in enumerate(features):
            name=fea_name
            value=li[fea_index[k]]
            if value!='':
                #value=int(float(value)*1000)
                try:
                    value=long(value)
                except:
                    value=round_to_n(float(value), 3)
            feats.append((k, name+"-"+str(value)))
        
        feats = gen_hashed_fm_feats(feats, int(1e+7))
        fout.write(li[lab_index] + ' ' + ' '.join(feats) + '\n')


fin.close()
fout.close()

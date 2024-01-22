import datetime as dt
from InfluxOPCEnv import InfluxOPCEnv

def date_to_timestamp_reseau(date):
    """
    Input: date string 
    Returns: timestamp for that date
    """
    
    # add leading 0 to hour
    time_list = date.split(' ')
    if time_list[1][1] == ':':
        time_list[1] = "0" + time_list[1]
    time_str = " ".join(time_list)
    
    formats = ['%Y-%m-%d %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p'] #valid formats in the dataset
    valid_format = False

    for format in formats:
        try:
            time = dt.datetime.strptime(time_str, format)
            valid_format   = True
            break
        except:
            pass
    assert valid_format
    return dt.datetime.timestamp(time)


class ReseauEnv(InfluxOPCEnv):
    def take_action(self, a):
        raise NotImplementedError
    
    def _get_reward(self, s, a):
        raise NotImplementedError
    
#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    error = [abs(x-y) for x,y in zip(predictions,net_worths)]
    temp_out = zip(ages,net_worths,error)

    import operator
    temp_out.sort(key = operator.itemgetter(2),reverse=True)
    #print(temp_out)
    cleaned_data = temp_out[10:]
         
    ### your code goes here

    
    return cleaned_data



from numpy import array


def merge(a,b):
    c = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    if i < len(a):
        c.extend(a[i:])
    if j < len(b):
        c.extend(b[j:])
    return c

print(merge([1,3,5,7,9],[2,4,6,8,10]))


def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

print(binary_search([1,2,3,4,5,6,7,8,9,10], 5))




def binary_search_first(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            if mid == 0 or arr[mid-1] != x:
                return mid
            else:
                high = mid - 1
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

print(binary_search_first([1,2,3,4,5,6,7,8,9,10], 5))   


def binary_search_first_last(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            if mid == 0 or arr[mid-1] != x:
                return mid
            else:
                high = mid - 1
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1   

    
print(binary_search_first_last([1,2,3,4,5,6,7,8,9,10], 5))


def binary_search_last(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            if mid == len(arr)-1 or arr[mid+1] != x:
                return mid
            else:
                low = mid + 1
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1


print(binary_search_last([1,2,3,4,5,6,7,8,9,10], 5))

#binary search with recursion
def binary_search_recursion(arr, x):
    if len(arr) == 0:
        return -1
    else:
        mid = len(arr) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            return binary_search_recursion(arr[mid+1:], x)
        else:
            return binary_search_recursion(arr[:mid], x)


print(binary_search_recursion([1,2,3,4,5,6,7,8,9,10], 5))

#binary search with recursion and first and last
def binary_search_recursion_first_last(arr, x):
    if len(arr) == 0:
        return -1
    else:
        mid = len(arr) // 2
        if arr[mid] == x:
            if mid == 0 or arr[mid-1] != x:
                return mid
            else:
                return binary_search_recursion_first_last(arr[mid+1:], x)
        elif arr[mid] < x:
            return binary_search_recursion_first_last(arr[mid+1:], x)
        else:
            return binary_search_recursion_first_last(arr[:mid], x)


print(binary_search_recursion_first_last([1,2,3,4,5,6,7,8,9,10], 5))

#binary search with recursion and first

def binary_search_recursion_first(arr, x):
    if len(arr) == 0:
        return -1
    else:
        mid = len(arr) // 2
        if arr[mid] == x:
            if mid == 0 or arr[mid-1] != x:
                return mid
            else:
                return binary_search_recursion_first(arr[mid+1:], x)
        elif arr[mid] < x:
            return binary_search_recursion_first(arr[mid+1:], x)
        else:
            return binary_search_recursion_first(arr[:mid], x)


print(binary_search_recursion_first([1,2,3,4,5,6,7,8,9,10], 5))

#binary search with recursion and last

def binary_search_recursion_last(arr, x):

    if len(arr) == 0:
        return -1
    else:
        mid = len(arr) // 2
        if arr[mid] == x:
            if mid == len(arr)-1 or arr[mid+1] != x:
                return mid
            else:
                return binary_search_recursion_last(arr[:mid], x)
        elif arr[mid] < x:
            return binary_search_recursion_last(arr[mid+1:], x)
        else:
            return binary_search_recursion_last(arr[:mid], x)


print(binary_search_recursion_last([1,2,3,4,5,6,7,8,9,10], 5))



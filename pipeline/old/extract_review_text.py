import os
import codecs
import json

data_directory = os.path.join('..', 'data','yelp_dataset')
businesses_filepath = os.path.join(data_directory,'business.json')
review_json_filepath = os.path.join(data_directory,'review.json')

intermediate_directory = os.path.join('..', 'intermediate')
review_txt_filepath = os.path.join(intermediate_directory,'review_text_all.txt')

# this is a bit time consuming
# This is what you swap out for news stories or whatever
def extract_review_text():

    restaurant_ids = set()

    # open the businesses file to get restaurant ids
    with codecs.open(businesses_filepath, encoding='utf_8') as f:
        # iterate through each line (json record) in the file
        for business_json in f:

            business = json.loads(business_json)
            # restaurants only
            if u'Restaurants' in business[u'categories']:
                restaurant_ids.add(business[u'business_id'])

    # turn restaurant_ids into a frozenset, as we don't need to change it anymore
    restaurant_ids = frozenset(restaurant_ids)

    # Write
    batch_size = 100
    review_count = 0

    # create & open a new file in write mode
    with codecs.open(review_txt_filepath, 'w', encoding='utf_8') as review_txt_file:

        # open the existing review json file
        with codecs.open(review_json_filepath, encoding='utf_8') as review_json_file:

            # loop through all reviews in the existing file and convert to dict
            batch = []
            for review_json in review_json_file:
                review = json.loads(review_json)

                # if this review is not about a restaurant, skip to the next one
                if review[u'business_id'] not in restaurant_ids:
                    continue

                batch.append(review[u'text'].replace('\n', '\\n') + '\n')
                review_count += 1
    
                if len(batch) == batch_size:
                    review_txt_file.write(''.join(batch))
                    batch = []

            # final write for partial patch
            review_txt_file.write(''.join(batch))

    print(u'''Text from {:,} restaurant reviews written to the new txt file.'''.format(review_count))


if not os.path.isfile(review_txt_filepath):
    extract_review_text()
else:
    print('Nope! File already exists...')

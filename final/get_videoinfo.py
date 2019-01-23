import requests as rq
from bs4 import BeautifulSoup
import numpy as np
import csv
import pandas as pd

channellist = ['Amateur', 'Anal', 'Arab', 'Asian', 'Babes', 'Babysitters', 'BBW', 'BDSM', 'Beach', 'BigBoobs', 'Bisexuals', 'BlackandEbony', 'BlackGays', 'Blondes', 'Blowjobs', 'Brazilian', 'British', 'Brunettes', 'Bukkake', 'Cartoons', 'Celebrities', 'Chinese', 'Close-ups', 'CreamPie', 'Cuckold', 'Cumshots', 'Czech', 'Danish', 'DoublePenetration', 'Emo', 'FaceSitting', 'Facials', 'Femdom', 'Fingering', 'Flashing', 'FootFetish', 'French', 'Funny', 'Gangbang', 'Gaping', 'Gays', 'German', 'Gothic', 'Grannies', 'GroupSex', 'Hairy', 'Handjobs', 'Hardcore', 'Hentai', 'HiddenCams', 'Indian', 'Interracial', 'Italian', 'Japanese', 'Korean', 'Ladyboys', 'Latex', 'Latin', 'Lesbians', 'Lingerie', 'Massage', 'Masturbation', 'Matures', 'Men', 'Midgets', 'MILFs', 'Nipples', 'Old+Young', 'Pornstars', 'POV', 'PublicNudity', 'Redheads', 'Russian', 'SexToys', 'Shemales', 'Showers', 'Softcore', 'Spanking', 'Squirting', 'Stockings', 'Strapon', 'Swedish', 'Swingers', 'Teens', 'Thai', 'Threesomes', 'Tits', 'Turkish', 'Upskirts', 'Vintage', 'Voyeur', 'Webcams']
print(len(channellist))
upload_date = []    #done
nb_votes = []       #done
nb_comments = []    #done
channels = []       #done
runtime = []        #done
nb_views = []       #done
for testidx in range(0,10):
    url = "https://xhamster.com/"
    response = rq.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    av = soup.findAll("a", {"class": "video-thumb-info__name"})
    cnt=0
    for i in av:
        if cnt==10:
            break
        perurl = i.get('href')
        perresponse = rq.get(perurl)
        persoup = BeautifulSoup(perresponse.text, "html.parser")
        print("i = ",cnt)

        data = persoup.findAll("meta",  itemprop="duration")
        if 'H' in data[0].get('content'):
            continue
        minute = int(data[0].get('content').split('T')[-1].split('M')[0])
        second = int(data[0].get('content').split('M')[-1].split('S')[0])
        duration = minute * 60 + second
        runtime += [duration]

        data = persoup.findAll("meta",  itemprop="interactionCount")
        idx = int(int(data[3].get('content').split(':')[-1])/10000)
        if idx >= 30:
            tmp=30
        else:
            tmp=idx
        nb_views += [tmp]

        like = int(data[1].get('content').split(':')[-1])
        unlike = int(data[2].get('content').split(':')[-1])
        nb_votes += [like-unlike]

        nb_comments += [data[0].get('content').split(':')[-1]]

        

        data = persoup.findAll("div",  itemprop="datePublished uploadDate")
        year = data[0].get('content').split('-')[0]
        upload_date += [year]

        data = persoup.findAll("meta",  itemprop="name")
        record = [0 for _ in range(len(channellist))]
        print(len(data))
        for idx in range(3,len(data)):
            # print("content = ",data[idx].get('content'))
            if data[idx].get('content') in channellist :
                channelidx = channellist.index(data[idx].get('content'))
                record[channelidx] = 1

        # print(record)
        channels +=[record]
        cnt+=1

print("========= below is target ============")
print(np.shape(nb_views))
print("========= below is feature ============")
print(np.shape(nb_votes))
print(np.shape(nb_comments))
print(np.shape(runtime))
print(np.shape(upload_date))
print(np.shape(channels))
print("========= end ============")

df = pd.DataFrame(np.array(upload_date), columns=['upload_date'])
df['nb_votes'] = np.array(nb_votes)
df['nb_comments'] = np.array(nb_comments)
df['runtime'] = np.array(runtime)
df['nb_views'] = np.array(nb_views)
print(df)
test_new_feature=np.array(np.array([channels]))
test_new_feature = test_new_feature[0]
new_test_all = np.concatenate((df[['upload_date','nb_votes', 'nb_comments','runtime']],test_new_feature), axis=1)

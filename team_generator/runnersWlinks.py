from Vlteamgenerator_2022_full import get_runners


RWL = "runnersWlinks.csv"
link = "http://192.168.2.135:8088/api/?Function=DataSourceSelectRow&Value=VLRunners,Runners2022,"

def main():
    runners, clubs = get_runners()
    with open(RWL, 'w', encoding='utf-8') as f:
        for runner in runners.values():
            f.write(f"{runner.name};{runner.club};{link}{runner.row}\n")


if __name__ == '__main__':
    main()
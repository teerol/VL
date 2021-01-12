import os,sys
import csv
tsvFile = "VL-RunnersRegister - Runners.tsv"
teamcsv_from_irma = "ilmoittautumiset.csv"
new_csv = "VLteams.csv" # tanne kirjoitetaan joukkuekortit
M_HTML = "VL_juoksijakortit_M.html"
N_HTML = "VL_juoksijakortit_N.html"

class Runner:
    def __init__(self, name, club, photofile, gender, row):
        self.name = name
        self.club = club
        self.photofile = photofile
        self.gender = gender
        self.row = row


def name_swap(name):
    l = name.split(' ')
    name = l[-1]
    del l[-1]
    for i in l:
        name += ' ' + i
    return name


def read_runners_and_teams(runnerdir, teamdir):
    club_runners = {}
    register = open(tsvFile, 'r', encoding='UTF-8')
    for row in register:
        row = row.split('\t')
        club = row[3]
        name = row[4]
        gender = row[2]
        register_row = row[0]
        try:
            runner_photo = row[14]
            club_photo = row[15]
            if os.path.isfile(runner_photo) and os.path.isfile(club_photo):
                r = Runner(name, club, runner_photo, gender, register_row)
                try:
                    runnerdir[club][name] = r
                except KeyError:
                    runnerdir[club] = {name : r}
                teamdir[club] = club_photo
        except IndexError:
            continue

def main():
    runnerdir = {}
    teamdir = {}
    read_runners_and_teams(runnerdir, teamdir)
    irmacsv = open(teamcsv_from_irma, 'r')
    rn = 1
    runners_found=[]
    extras = []
    with open(new_csv, 'w') as NEW_csv:
        with open(M_HTML, 'w') as MHTML:
            with open(N_HTML, 'w') as NHTML:
                writer = csv.writer(NEW_csv, delimiter=';', lineterminator='\n')
                htmlstart = "<html><head><style>tr:nth-child(2n) {background:#D0E4F5; }</style></head><body><table><tr><th>Joukkuekortti</th><th>Juoksijakortit</th></tr>"
                MHTML.write(htmlstart)
                NHTML.write(htmlstart)
                for team in irmacsv:
                    team = team.split(';')
                    del team[-1] # deleting newline
                    Class = team[0]
                    team_name = team[1]
                    club_name = team[2]
                    del team[0:3]
                    newline="<tr><td><a target='iframe' href ='http://192.168.2.135:8088/api/?Function=DataSourceSelectRow&Value=VLRunners,Teams,"+ str(rn)+"'>"+team_name+"</a></td>"
                    try:
                        club_photo = teamdir[club_name]
                    except KeyError:
                        print(club_name," KeyError, club")
                        continue
                    row = [rn, club_name, team_name, club_photo]
                    runner_photos = []
                    for runner in team:
                        runner = name_swap(runner)
                        try:
                            runner_object = runnerdir[club_name][runner]
                            row.append(runner_object.name)
                            runner_photos.append(runner_object.photofile)
                            runners_found.append(runner)
                            newline+="<td><a target='iframe' href ='http://192.168.2.135:8088/api/?Function=DataSourceSelectRow&Value=VLRunners,Runners,"+ str(runner_object.row)+"'>"+runner+"</a></td>"
                        except KeyError:
                            print(runner, " KeyError, runner")
                            pass
                    if len(runner_photos) == len(team):
                        row += runner_photos
                        # print(row)
                        writer.writerow(row)
                        rn += 1
                        newline += "</tr>\n"
                        if Class == "H21":
                            MHTML.write(newline)
                        elif Class == "D21":
                            NHTML.write(newline)
                    else:
                        for p in team:
                            extras.append(name_swap(p))

                # adding runners those are not in teams
                head = "<tr><th>EI joukkueissa:</th></tr>\n"
                MHTML.write(head)
                NHTML.write(head)
                for club in runnerdir:
                    clubo=runnerdir[club]
                    for runnere in clubo:
                        runnerobj=runnerdir[club][runnere]
                        if runnere in runners_found and runnere in extras:
                            sline = "<td>"+runnerobj.club+"</td><td><a target='iframe' href ='http://192.168.2.135:8088/api/?Function=DataSourceSelectRow&Value=VLRunners,Runners,"+ str(runnerobj.row)+"'>"+runnere+"</a></td></tr>\n"
                            if runnerobj.gender == "M":
                                MHTML.write(sline)
                            elif runnerobj.gender == "N":
                                NHTML.write(sline)
                htmlend = "</table></body></html>"
                NHTML.write(htmlend)
                MHTML.write(htmlend)


main()

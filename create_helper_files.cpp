#include <bits/stdc++.h>
#include <sys/types.h>
#include <dirent.h>

using namespace std;

// Number of frames per action used in training
#define SAMPLE_SIZE 10

map<string,int> scenario_to_id;
int id_count=0;

// Parse Charades' csv files and create lists of samples for each class
// csv_filename: name of the input file
// preffix: preffix for the output files ("prefix%d.txt")
void create_helper_files(string csv_filename, string preffix) {
	// specify what csv columns to use
	// check http://vuchallenge.org/README-charades.txt for details
	vector<int> items;
	items.push_back(0); // id
	items.push_back(2); // scenario

	// open csv file for reading
	ifstream in;
	in.open(csv_filename, ifstream::in);
	string s;
	getline(in,s);
	while(getline(in,s)) {
		// parse selected columns from an input line
		vector<string> v;
		for(int k=0; k < items.size(); k++) {
			int i = items[k];
			string out="";
			int j=0, p=0;
			while(j < i) {
				while(p < s.size() && s[p] != ',') {
					if(s[p] == '"') {
						p++;
						while(p < s.size() && s[p] != '"')
							p++;
					}
					p++;
				}
				p++;
				j++;
			}
			if(j==i) {
				while(p < s.size() && s[p] != ',') {
					out += s[p];
					p++;
				}
			}
			v.push_back(out);
		}

		// discover scenario identifier
		if(scenario_to_id.count(v[1]) == 0)
			scenario_to_id[v[1]] = id_count++;
		int id = scenario_to_id[v[1]];

		// discover the name of available frames for the current video
		vector<string> frames;
		DIR *dir;
		struct dirent *ent;
		string dirname = "Charades_v1_rgb_scaled/";
		dirname += v[0]+"/";
		dir = opendir(dirname.c_str());
		while((ent = readdir(dir)) != NULL)
			if(ent->d_name[0] != '.')
				frames.push_back(ent->d_name);
		closedir(dir);

		// output filename
		string outfile = preffix;
		outfile += to_string(id);
		outfile += ".txt";

		// save frame names
		ofstream fp;
		fp.open(outfile, ofstream::out | ofstream::app);
		for(string f : frames)
			fp << "Charades_v1_rgb_scaled/" << v[0] << "/" << f << endl;
		fp.close();
	}
	in.close();
}

// Main program
int main(int argc, char **argv) {
	// Create training helper files
	create_helper_files("vu17_charades/Charades_vu17_train.csv", "helper_files/train_");

	// Create validation helper files
	create_helper_files("vu17_charades/Charades_vu17_validation.csv", "helper_files/val_");

	// Save the name of the classes
	ofstream fp;
	fp.open("scenario_id.txt", ofstream::out | ofstream::app);
	for(auto it = scenario_to_id.begin(); it != scenario_to_id.end(); it++)
		fp << it->second << " " << it->first << endl;
	fp.close();

	return 0;
}

#include <bits/stdc++.h>
#include <sys/types.h>
#include <dirent.h>

using namespace std;

// Number of frames per action used in training
#define SAMPLE_SIZE 10

// Parse Charades' csv files and create lists of samples for each class
// csv_filename: name of the input file
// preffix: preffix for the output files ("prefix%d.txt")
void create_helper_files(string csv_filename, string preffix) {
	// specify what csv columns to use
	// check http://vuchallenge.org/README-charades.txt for details
	vector<int> items;
	items.push_back(0); // id
	items.push_back(9); // actions

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

		// discover the number of available frames for the current video
		DIR *dir;
		struct dirent *ent;
		string dirname = "Charades_v1_features_flow/";
		dirname += v[0]+"/";
		dir = opendir(dirname.c_str());
		s="";
		while((ent = readdir(dir)) != NULL) {
			string tmp = ent->d_name;
			if(tmp > s)
				s = tmp;
		}
		closedir(dir);
		int numframes = stoi(s.substr(s.find("-")+1, s.find(".")), nullptr, 10);

		// parse actions
		s = v[1];
		while(s.size()) {
			// get next action
			string tmp;
			int pos = s.find(";");
			if(pos >= 0) {
				tmp = s.substr(0, pos);
				s = s.substr(pos+1, s.size());
			}
			else {
				tmp = s;
				s = "";
			}
			stringstream stream(tmp.substr(1, tmp.size()));
			int action; // class number
			double start, end; // time interval in seconds
			stream >> action >> start >> end;

			// output filename
			string outfile = preffix;
			outfile += to_string(action);
			outfile += ".txt";

			// check if time interval is valid
			if(end <= start)
				continue;

			// convert time interval to frame number (24 fps)
			start = 1.0+start*24.0;
			end = 1.0+end*24.0;
			int i, j;
			for(i=1; i < start; i+=4);
			if(i >= numframes)
				continue;
			for(j=i; j+4 < end && j+4 <= numframes; j+=4);
			vector<int> vi;
			for(int k=i; k <= j; k+=4)
				vi.push_back(k);

			// pick SAMPLE_SIZE frames and save their names
			ofstream fp;
			fp.open(outfile, ofstream::out | ofstream::app);
			for(int k=0; k < SAMPLE_SIZE; k++)
				fp << "Charades_v1_features_flow/" << v[0] << "/" << v[0] << "-" << setfill('0') << setw(6) << vi[((vi.size()-1)*k) / (SAMPLE_SIZE-1)] << ".txt" << (k==(SAMPLE_SIZE-1)?"":";");
			fp << endl;
			fp.close();
		};
	}
	in.close();
}

// Main program
int main(int argc, char **argv) {
	// Create training helper files
	create_helper_files("vu17_charades/Charades_vu17_train.csv", "helper_files/train_");

	// Create validation helper files
	create_helper_files("vu17_charades/Charades_vu17_validation.csv", "helper_files/val_");

	return 0;
}


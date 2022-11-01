#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

typedef pair<vector<int>::iterator,vector<int>::iterator> Neighbors;

class Graph{
	public:
		vector<int> sources;
		vector<int> targets;
		int num_vertices;
		vector<vector<int>> lprops;
		vector<vector<float>> fprops;
		vector<int> edge_starts;

		Graph(){}
		Graph(int num_verts):num_vertices(num_verts){
			edge_starts = vector<int>(num_vertices+1,0);
			sources.reserve(num_vertices*6); // just some heuristic estimate on how many edges there will be
			targets.reserve(num_vertices*6);
		}

		bool add_edge_onside(int s, int t){
			// Does not add edges if already exists
			int ps,pt;
			vector<int>::iterator ps_pointer = edge_starts.begin()+s;
			ps = *ps_pointer;
			int next_edges = *(ps_pointer+1);
			while(ps!=next_edges && targets[ps]<t){
				ps++;
			}
			if (ps!=next_edges && targets[ps] == t){
				return false;
			}
			sources.insert(sources.begin()+ps,s);
			targets.insert(targets.begin()+ps,t);
			++ps_pointer;
			for (;ps_pointer!=edge_starts.end();++ps_pointer){
			  (*ps_pointer)+=1;
			}
			return true;
		}

		bool add_edge(int v1, int v2){
			// Does not add edges if already exists
			if (add_edge_onside(v1,v2)){
				return add_edge_onside(v2,v1);
			}
			return false;
		}

		Neighbors adjacent_vertices(int vertex){
			vector<int>::iterator edge_start = edge_starts.begin()+vertex;
			return Neighbors(targets.begin()+(*edge_start),targets.begin()+*(edge_start+1));
		}

		int num_neighbors(int vertex){
			vector<int>::iterator edge_start = edge_starts.begin()+vertex;
			return *(edge_start+1)-*edge_start;

		}
		
		bool delete_edge(int v1, int v2){
			return delete_edge_onesided(v1,v2)&&delete_edge_onesided(v2,v1);
		}

		bool delete_edge_onesided(int source, int target){
			int targ_vert;
			int pointer;
			vector<int>::iterator edge_pointer, pb_pointer;
			edge_pointer = edge_starts.begin()+source;
			pointer = *edge_pointer;
			int next_edges = *(edge_pointer+1);
			while (pointer!=next_edges){
				targ_vert = targets[pointer];
				if (targ_vert==target){
					sources.erase(sources.begin()+pointer);
					targets.erase(targets.begin()+pointer);
					for (pb_pointer = edge_pointer+1;pb_pointer!=edge_starts.end();pb_pointer++){
						(*pb_pointer)-=1;
					}
					return true;
				}
				pointer++;
			}
			return false;
		}

		void delete_many_onesided(vector<int> s, int t){
			// assume sources sorted ascending
			vector<int>::iterator source_pointer;
			int cur;
			source_pointer = s.begin();
			int ind = 0;
			int decreaso = 0;
			for (vector<int>::iterator p = edge_starts.begin();p!=edge_starts.end();++p,++ind){
				(*p)-=decreaso;
				if (source_pointer!=s.end() && ind == *source_pointer){
					int cur=*p;
					while (targets[cur]!=t){ // search for target
						assert(sources[cur]==*source_pointer);
						++cur;
					}
					sources.erase(sources.begin()+cur);
					targets.erase(targets.begin()+cur);
					source_pointer++;
					decreaso++;
				}
			}
		}

		void add_lprop(int init){
			lprops.push_back(vector<int>(num_vertices,init));
		}

		void add_fprop(float init){
			fprops.push_back(vector<float>(num_vertices,init));
		}

		bool edge_exists(int source, int target){
			vector<int>::iterator p = edge_starts.begin()+source;
			int begin = *p;
			vector<int>::iterator next = targets.begin()+*(p+1);
			for (vector<int>::iterator t = targets.begin()+begin;t!=next;++t){
				if (*t==target){
					return true;
				}
			}
			return false;
		}

		void remove_vertex(int vertex){
			if (vertex != num_vertices-1){
				// Assumes that the vertex is already cleared. Swaps highest index vertex into this vertex position
				int edge_loc = *(edge_starts.end()-2); //-2, because last elem contains end pointers
				vector<int> final_targets(targets.begin()+edge_loc,targets.end()); 
				sources.erase(sources.begin()+edge_loc,sources.end());
				targets.erase(targets.begin()+edge_loc,targets.end());
				
				vector<int>::iterator edge_point = edge_starts.begin()+vertex;
				edge_loc = *edge_point;
				targets.insert(targets.begin()+edge_loc,final_targets.begin(),final_targets.end());
				sources.insert(sources.begin()+edge_loc,final_targets.size(),vertex);
				++edge_point;
				for (;edge_point!=edge_starts.end()-1;++edge_point){
					(*edge_point)+=final_targets.size();
				}
				for (int target : final_targets){
					edge_point = edge_starts.begin()+target;
					edge_loc = *edge_point;
					int next_loc = *(edge_point+1);
					vector<int>::iterator targ_it = targets.begin()+(next_loc-1);
					assert(*targ_it =num_vertices-1);
					targets.erase(targ_it);
					sources.erase(sources.begin()+(next_loc-1));
					while(edge_loc!=next_loc-1 && targets[edge_loc]<vertex){
						edge_loc++;
					}
					sources.insert(sources.begin()+edge_loc,target);
					targets.insert(targets.begin()+edge_loc,vertex);
				}
			}
			num_vertices--;
			edge_starts.erase(edge_starts.end()-2);
			for (vector<vector<int>>::iterator vp = lprops.begin();vp!=lprops.end();++vp){
				(*vp)[vertex] = *(vp->end()-1);
				vp->pop_back();
			}
			for (vector<vector<float>>::iterator vp = fprops.begin();vp!=fprops.end();++vp){
				(*vp)[vertex] = *(vp->end()-1);
				vp->pop_back();
			}
		}

		void clear_vertex(int vertex){
			Neighbors neigh = adjacent_vertices(vertex);
			vector<int> del_sources(neigh.first,neigh.second);
			delete_many_onesided(del_sources,vertex);

			vector<int>::iterator edge_pointer, pb_pointer;
			edge_pointer = edge_starts.begin()+vertex;
			int next_edges = *(edge_pointer+1);
			
			int cur_edges = *edge_pointer;
			int num_del = next_edges - cur_edges;
			sources.erase(sources.begin()+cur_edges,sources.begin()+next_edges);
			targets.erase(targets.begin()+cur_edges,targets.begin()+next_edges);
			for (pb_pointer = edge_pointer+1;pb_pointer!=edge_starts.end();pb_pointer++){
				(*pb_pointer)-=num_del;
			}
		}

		void do_complete_dump(string fname="graph_dump.txt"){
			vector<int>::iterator s,t,v;
			vector<float>::iterator f;
			vector<int>::iterator p;
			vector<vector<int>>::iterator lp;
			vector<vector<float>>::iterator fp;
			ofstream my_file;
			my_file.open(fname);
			my_file << "num vertices: " << num_vertices << endl << endl;
			my_file << "Edges:" << endl;
			for (s=sources.begin(),t=targets.begin();s!=sources.end();++s,++t){
				my_file << to_string(*s) << "\t" << to_string(*t) << endl;
			}
			my_file << endl << "Edge Pointers:" << endl;
			for (p=edge_starts.begin();p!=edge_starts.end();p++){
				my_file << *p << " ";
			}
			my_file << endl;

			my_file << endl << "Long Properties:" << endl;
			for (lp=lprops.begin();lp!=lprops.end();lp++){
				for (v=lp->begin();v!=lp->end();v++){
					my_file << *v << " ";
				}
				my_file << endl << endl;
			}
			my_file << endl << "Long Properties:" << endl;
			for (fp=fprops.begin();fp!=fprops.end();fp++){
				for (f=fp->begin();f!=fp->end();f++){
					my_file << *f << " ";
				}
				my_file << endl << endl;
			}
			my_file.close();
		}
		void graphviz_me(vector<pair<string,vector<string>>> props, string fname="my_graph.dot", bool undirected=true){
			ofstream my_file;
			vector<int>::iterator s,t;
			my_file.open(fname);
			my_file << "graph G {" << endl;
			for (int i=0;i<num_vertices;i++){
				my_file << to_string(i) << " [";
				for (auto p=props.begin();p!=props.end();++p){
					my_file<<p->first<<"=\"";
					my_file<<p->second[i]<<"\"";
					if (p!=props.end()-1){
						my_file << ", ";
					}
				}
				my_file<<"];"<<endl;
			}
			for (s=sources.begin(),t=targets.begin();s!=sources.end();++s,++t){
				if (!undirected || (*s<*t)){
					my_file << to_string(*s) << "--" << to_string(*t) << " ;" << endl;
				}
			}
			my_file<<"}";
			my_file.close();
		}

		void graphviz_me(string fname="my_graph.dot",bool undirected=true){
			ofstream my_file;
			vector<int>::iterator s,t;
			my_file.open(fname);
			my_file << "graph G {" << endl;
			for (int i=0;i<num_vertices;i++){
				my_file << to_string(i) << " ;" << endl;
			}
			for (s=sources.begin(),t=targets.begin();s!=sources.end();++s,++t){
				if (!undirected || (*s<*t)){
					my_file << to_string(*s) << "--" << to_string(*t) << " ;" << endl;
				}
			}
			my_file<<"}";
			my_file.close();
		}
};

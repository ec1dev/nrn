#include <string>

#include <music.hh>

#include <unordered_map>

class PreSyn;
using Gi2PreSynTable = std::unordered_map<int, PreSyn*>;

// Interface which nrnmusic.so module (which mdj writes) will use

namespace NRNMUSIC {

class EventOutputPort: public MUSIC::EventOutputPort {
  public:
    EventOutputPort(MUSIC::Setup* s, std::string id)
        : MUSIC::Port(s, id)
        , MUSIC::EventOutputPort(s, id) {}

    // Connect a gid to a MUSIC global index in this port
    void gid2index(int gid, int gi);

    Gi2PreSynTable* gi_table;
};

class EventInputPort: public MUSIC::EventInputPort {
  public:
    EventInputPort(MUSIC::Setup* s, std::string id);

    // Connect a MUSIC global index to a point process: the new NetCon, with a reference for the
    // caller (neuronmusic.pyx gives both as the interface's Object addresses).
    void* index2target(int gi, void* target);

    Gi2PreSynTable* gi_table;
};

EventOutputPort* publishEventOutput(std::string identifier);

EventInputPort* publishEventInput(std::string identifier);

}  // namespace NRNMUSIC

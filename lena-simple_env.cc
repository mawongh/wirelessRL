/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Manuel Wong <mawongh@gmail.com>
 * Based on the example provided by Jaume Nin <jnin@cttc.es>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/config-store.h"
#include <ns3/buildings-propagation-loss-model.h>
#include <ns3/buildings-helper.h>
#include <ns3/radio-environment-map-helper.h>
#include <iomanip>
#include <string>
#include <vector>
// additional libraries
#include <iostream>
#include <fstream>
#include <sstream>

using namespace ns3;
using std::vector;

//Defining and initialing global variables
std::string input_file = "scratch/lena-simple_env_temp.csv";
uint32_t nEnb;
vector < int > siteid;
vector < int > nodeid;
vector < int > lon;
vector < int > lat;
vector < int > height;
vector < int > azimuth;
vector < int > beamwidth;
vector < int > txpower;
vector < int > gain;

//This function reads the contents of the configuration file and fills out the global variables
int read_conf() {
    std::string line;
    std::ifstream file;
    file.open(input_file);
    int f1, f2, f3, f4, f5, f6, f7, f8, f9;
    char first_char;
    while (getline(file, line)) {
        first_char = line.at(0);
        if (first_char != '#') {
            std::istringstream ss(line);
            char c1, c2, c3, c4, c5, c6, c7, c8;
        
            ss >> f1 >> c1 >>
                  f2 >> c2 >>
                  f3 >> c3 >>
                  f4 >> c4 >>
                  f5 >> c5 >>
                  f6 >> c6 >>
                  f7 >> c7 >>
                  f8 >> c8 >>
                  f9;
            siteid.push_back(f1);
            nodeid.push_back(f2);
            lon.push_back(f3);
            lat.push_back(f4);
            height.push_back(f5);
            azimuth.push_back(f6);
            beamwidth.push_back(f7);
            txpower.push_back(f8);
            gain.push_back(f9);
            
            nEnb++;
        }
    }
    file.close();
    return 0;
} // end of the read_conf() function



int
main (int argc, char *argv[]) {
    CommandLine cmd;
    cmd.Parse (argc, argv);

//    Reads the configuration file
    read_conf();
    
//    Variables initialization
    uint32_t nUe = 1; // Number of UEs per Node (sector)
    double nodeHeight = 1.5; // Height in meters of the UEs
    double roomLength = 1000; // Size of the square around the node where the UE will be positioned

    ConfigStore inputConfig;
    inputConfig.ConfigureDefaults ();

    cmd.Parse (argc, argv);

    Ptr < LteHelper > lteHelper = CreateObject<LteHelper> ();
    // lteHelper->EnableLogComponents ();
    lteHelper->SetAttribute ("PathlossModel", StringValue ("ns3::FriisPropagationLossModel"));

//   Create Nodes: eNodeB and UE
    NodeContainer enbNodes;
    NodeContainer threeSectorNodes;
    vector < NodeContainer > ueNodes;


    threeSectorNodes.Create (nEnb);
    enbNodes.Add (threeSectorNodes);

    for (uint32_t i = 0; i < nEnb; i++) {
        NodeContainer ueNode;
        ueNode.Create (nUe);
        ueNodes.push_back (ueNode);
        }

    MobilityHelper mobility;
    vector<Vector> enbPosition;
    Ptr < ListPositionAllocator > positionAlloc = CreateObject<ListPositionAllocator> ();
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (enbNodes);
    BuildingsHelper::Install (enbNodes);
    
    // Add each sector
    for (uint32_t index = 0; index < nEnb; index++) {

        Vector v (lon.at(index), lat.at(index), height.at(index));
        positionAlloc->Add (v);
        enbPosition.push_back (v);
        mobility.Install (ueNodes.at(index));
    }
  
  mobility.SetPositionAllocator (positionAlloc);
  mobility.Install (enbNodes);

  // Position of UEs attached to eNB
  for (uint32_t i = 0; i < nEnb; i++) {
        Ptr<UniformRandomVariable> posX = CreateObject<UniformRandomVariable> ();
        posX->SetAttribute ("Min", DoubleValue (enbPosition.at(i).x - roomLength * 0.5));
        posX->SetAttribute ("Max", DoubleValue (enbPosition.at(i).x + roomLength * 0.5));
        Ptr<UniformRandomVariable> posY = CreateObject<UniformRandomVariable> ();
        posY->SetAttribute ("Min", DoubleValue (enbPosition.at(i).y - roomLength * 0.5));
        posY->SetAttribute ("Max", DoubleValue (enbPosition.at(i).y + roomLength * 0.5));
        positionAlloc = CreateObject<ListPositionAllocator> ();
      
        for (uint32_t j = 0; j < nUe; j++) {
            positionAlloc->Add (Vector (posX->GetValue (), posY->GetValue (), nodeHeight));
            mobility.SetPositionAllocator (positionAlloc);
            }
      
        mobility.Install (ueNodes.at(i));
        BuildingsHelper::Install (ueNodes.at(i));
    }

//   Create Devices and install them in the Nodes (eNB and UE)
    NetDeviceContainer enbDevs;
    vector < NetDeviceContainer > ueDevs;

    for (uint32_t index = 0; index < nEnb; index++) {
        Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (txpower.at(index)));
        lteHelper->SetEnbAntennaModelType ("ns3::CosineAntennaModel");
        lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (azimuth.at(index)));
        lteHelper->SetEnbAntennaModelAttribute ("Beamwidth",   DoubleValue (beamwidth.at(index)));
        lteHelper->SetEnbAntennaModelAttribute ("MaxGain",     DoubleValue (gain.at(index)));
        enbDevs.Add ( lteHelper->InstallEnbDevice (threeSectorNodes.Get (index)));
    }
    
    
    for (uint32_t i = 0; i < nEnb; i++) {
        NetDeviceContainer ueDev = lteHelper->InstallUeDevice (ueNodes.at(i));
        ueDevs.push_back (ueDev);
        lteHelper->Attach (ueDev, enbDevs.Get (i));
        enum EpsBearer::Qci q = EpsBearer::GBR_CONV_VOICE;
        EpsBearer bearer (q);
        lteHelper->ActivateDataRadioBearer (ueDev, bearer);
        }

//   by default, simulation will anyway stop right after the REM has been generated
    Simulator::Stop (Seconds (0.0069));

    Ptr<RadioEnvironmentMapHelper> remHelper = CreateObject<RadioEnvironmentMapHelper> ();
    remHelper->SetAttribute ("ChannelPath", StringValue ("/ChannelList/0"));
    remHelper->SetAttribute ("OutputFile", StringValue ("scratch/lena-simple_env.rem"));
    remHelper->SetAttribute ("XMin", DoubleValue (300.0));
    remHelper->SetAttribute ("XMax", DoubleValue (+3300.0));
    remHelper->SetAttribute ("YMin", DoubleValue (0.0));
    remHelper->SetAttribute ("YMax", DoubleValue (+5000.0));
    remHelper->SetAttribute ("Z", DoubleValue (1.5));
    // remHelper->SetAttribute ("UseDataChannel", BooleanValue (true));
    remHelper->Install ();

    Simulator::Run ();

//  GtkConfigStore config;
//  config.ConfigureAttributes ();

    lteHelper = 0;
    Simulator::Destroy ();
    return 0;
}



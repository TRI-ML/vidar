
parallel_domain_to_cityscapes = {
    # ROAD
    8:    7,    # CrossWalk
    11:   7,    # LaneMarking
    12:   7,    # LimitLine
    15:   7,    # OtherDriveableSurface
    24:   7,    # Road
    27:   7,    # RoadMarking
    30:   7,    # TemporaryConstructionObject
    # SIDEWALK
    28:   8,    # SideWalk
    26:   8,    # RoadBoundary(Curb)
    16:   8,    # OtherFixedStructure
    17:   8,    # OtherMovable
    # WALL
    19:  12,    # Overpass / Bridge / Tunnel
    25:  12,    # RoadBarriers
    # FENCE
    9:   13,    # Fence
    # BUILDING
    3:   11,    # Building
    # POLE
    10:  17,    # HorizontalPole
    21:  17,    # ParkingMeter
    38:  17,    # VerticalPole
    # TRAFFIC LIGHT
    33:  19,    # TrafficLight
    # TRAFFIC SIGN
    34:  20,    # TrafficSign
    # VEGETATION
    37:  21,    # Vegetation
    # TERRAIN
    31:  22,    # Terrain
    # SKY
    29:  23,    # Sky
    # PERSON
    22:  24,    # Pedestrian
    # RIDER
    2:   25,    # Bicyclist
    14:  25,    # Motorcyclist
    18:  25,    # OtherRider
    # CAR
    5:   26,    # Car
    # TRUCK
    36:  27,    # Truck
    6:   27,    # Caravan/RV
    7:   27,    # ConstructionVehicle
    # BUS
    4:   28,    # Bus
    # TRAIN
    35:  31,    # Train
    # MOTORCYCLE
    13:  32,    # Motorcycle
    # BICYCLE
    1:   33,    # Bicycle
    # IGNORE
    0:  255,    # Animal
    20: 255,    # OwnCar(EgoCar)
    23: 255,    # Railway
    32: 255,    # TowedObject
    39: 255,    # WheeledSlow
    40: 255,    # Void
}

ddad_to_cityscapes = {
    # ROAD
    7:    7,    # Crosswalk
    10:   7,    # LaneMarking
    11:   7,    # LimitLine
    13:   7,    # OtherDriveableSurface
    21:   7,    # Road
    24:   7,    # RoadMarking
    27:   7,    # TemporaryConstructionObject
    # SIDEWALK
    25:   8,    # SideWalk
    23:   8,    # RoadBoundary (Curb)
    14:   8,    # OtherFixedStructure
    15:   8,    # OtherMovable
    # WALL
    16:  12,    # Overpass/Bridge/Tunnel
    22:  12,    # RoadBarriers
    # FENCE
    8:   13,    # Fence
    # BUILDING
    2:   11,    # Building
    # POLE
    9:   17,    # HorizontalPole
    35:  17,    # VerticalPole
    # TRAFFIC LIGHT
    30:  19,    # TrafficLight
    # TRAFFIC SIGN
    31:  20,    # TrafficSign
    # VEGETATION
    34:  21,    # Vegetation
    # TERRAIN
    28:  22,    # Terrain
    # SKY
    26:  23,    # Sky
    # PERSON
    18:  24,    # Pedestrian
    # RIDER
    20:  25,    # Rider
    # CAR
    4:   26,    # Car
    # TRUCK
    33:  27,    # Truck
    5:   27,    # Caravan/RV
    6:   27,    # ConstructionVehicle
    # BUS
    3:   28,    # Bus
    # TRAIN
    32:  31,    # Train
    # MOTORCYCLE
    12:  32,    # Motorcycle
    # BICYCLE
    1:   33,    # Bicycle
    # IGNORE
    0:  255,    # Animal
    17: 255,    # OwnCar (EgoCar)
    19: 255,    # Railway
    29: 255,    # TowedObject
    36: 255,    # WheeledSlow
    37: 255,    # Void
}

vkitti2_to_cityscapes = {
    # TERRAIN
    0:   22,    # Terrain
    # SKY
    1:   23,    # Sky
    # VEGETATION
    2:   21,    # Tree
    3:   21,    # Vegetation
    # BUILDING
    4:   11,    # Building
    # ROAD
    5:    7,    # Road
    # TRAFFIC SIGN
    7:   20,    # TrafficSign
    # TRAFFIC LIGHT
    8:   19,    # TrafficLight
    # POLE
    9:   17,    # Pole
    # CAR
    12:  26,    # Car
    # TRUCK
    11:  27,    # Truck
    13:  27,    # Van
    # IGNORE
    10: 255,    # Misc
    6:  255,    # GuardRail
    14: 255,    # Undefined
}

vkitti2_to_parallel_domain = {
    0:  31,     # Terrain
    1:  29,     # Sky
    2:  37,     # Tree
    3:  37,     # Vegetation
    4:   3,     # Building
    5:  24,     # Road
    6:  23,     # GuardRail
    7:  34,     # TrafficSign
    8:  33,     # TrafficLight
    9:  38,     # Pole
    10: 30,     # Misc
    11: 36,     # Truck
    12:  5,     # Car
    13:  6,     # Van
    14: 40,     # Undefined
}

ddad_to_parallel_domain = {
    0:   0,     # Animal
    1:   1,     # Bicycle
    2:   3,     # Building
    3:   4,     # Bus
    4:   5,     # Car
    5:   6,     # Caravan / RV
    6:   7,     # ConstructionVehicle
    7:   8,     # CrossWalk
    8:   9,     # Fence
    9:  10,     # HorizontalPole
    10: 11,     # LaneMarking
    11: 12,     # LimitLine
    12: 13,     # Motorcycle
    13: 15,     # OtherDrivableSurface
    14: 16,     # OtherFixedStructure
    15: 17,     # OtherMovable
    16: 19,     # Overpass / Bridge / Tunnel
    17: 20,     # OwnCar(EgoCar)
    18: 22,     # Pedestrian
    19: 23,     # Railway
    20: 14,     # Rider
    21: 24,     # Road
    22: 25,     # RoadBarriers
    23: 26,     # RoadBoundary(Curb)
    24: 27,     # RoadMarking
    25: 28,     # SideWalk
    26: 29,     # Sky
    27: 30,     # TemporaryConstructionObject
    28: 31,     # Terrain
    29: 32,     # TowedObject
    30: 33,     # TrafficLight
    31: 34,     # TrafficSign
    32: 35,     # Train
    33: 36,     # Truck
    34: 37,     # Vegetation
    35: 38,     # VerticalPole
    36: 39,     # WheeledSlow
    37: 40,     # Void
}

cityscapes_to_vkitti2 = {
    0:   14,    # unlabeled
    1:   14,    # ego vehicle
    2:   14,    # rectification border
    3:   14,    # out of roi
    4:   14,    # static
    5:   14,    # dynamic
    6:   14,    # ground
    7:    5,    # road
    8:   14,    # sidewalk
    9:   14,    # parking
    10:   6,    # rail track
    11:   4,    # building
    12:  14,    # wall
    13:  14,    # fence
    14:  14,    # guard rail
    15:  14,    # bridge
    16:  14,    # tunnel
    17:   9,    # pole
    18:  14,    # polegroup
    19:   8,    # traffic light
    20:   7,    # traffic sign
    21:   3,    # vegetation
    22:   0,    # terrain
    23:   1,    # sky
    24:  14,    # person
    25:  14,    # rider
    26:  12,    # car
    27:  11,    # truck
    28:  14,    # bus
    29:  13,    # caravan
    30:  14,    # trailer
    31:  14,    # train
    32:  14,    # motorcycle
    33:  14,    # bicycle
    34:  14,    # license plate
    255: 14,    # ignore
}

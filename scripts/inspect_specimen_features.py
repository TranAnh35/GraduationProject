import json

def main():
    with open('data/ground_truth/specimen_mask_features.json') as f:
        specs = json.load(f)

    for s in specs:
        key = s['key']
        print(f"=== SPECIMEN: {key} ===")
        features = s['features']
        for feat in features:
            fid = feat.get('id')
            kind = feat.get('kind')
            x = feat.get('x')
            y = feat.get('y')
            cx = feat.get('corrosionX')
            cy = feat.get('corrosionY')
            d = feat.get('diameter')
            dp = feat.get('depth')
            rd = feat.get('rivetDiameter')
            off = feat.get('offset')
            if key == 'rivet':
                if cx != x or cy != y:
                    print(f"  feat {fid}: x={x}, y={y}, cx={cx}, cy={cy}, offset={off}")
        if key == 'rivet':
            print("  (If nothing above, cx==x and cy==y for all rivet features)")

        if key == 'mixed':
            print("  Sample mixed offsets:")
            for feat in features[:5]:
                print(f"  feat {feat['id']}: kind={feat['kind']}, x={feat['x']}, y={feat['y']}, cx={feat['corrosionX']}, cy={feat['corrosionY']}, off={feat['offset']}, diam={feat['diameter']}")

if __name__ == '__main__':
    main()

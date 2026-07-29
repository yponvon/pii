# SG PII Reference Bank

Seed values for generating realistic synthetic call-transcript examples.
Pulled from user-provided reference (2026-07-14). Use these as raw material
to embed into messy, disfluent transcript text -- never insert them verbatim
as clean strings; always run them through the same "real transcription"
messiness (spelled out, interrupted, restated in a different format,
STT phonetic misheard forms, etc.) established in synthetic_example_01.json.

## SG_NRIC_FIN (prefix generations, masking, STT phonetic variants)

1. Standard Citizen (Pre-2000): S1234567D
2. Standard Citizen (Post-2000): T0123456B
3. Standard Foreigner (Pre-2000): F1234567N
4. Standard Foreigner (Post-2000): G0123456P
5. Standard Foreigner (Post-2022): M1234567K
6. Masked/Redacted: SXXXX567D
7. Spaced/Dashed Format: S-1234567-D, S 1234567 D
8. Lowercase (raw user input): s1234567d, t0123456b
9. Bracketed checksum: S1234567(D)
10. STT transcript phonetic: "s one two three four five six seven dog"
11. STT transcript mixed: "ic is tee zero one two three four five six boy"

Note: real gold data in data_all/ only ever tags the LAST FOUR characters
(e.g. "840D"), spoken after the agent asks "last four characters of your
NRIC" -- full 9-character NRIC is rare/absent in the real corpus. Match that
convention unless deliberately testing full-NRIC recognition.

## SG_PHONE_NUMBER (formats, quirks)

1. Standard Mobile (8-digit): 91234567
2. Standard Mobile Alternate: 81234567
3. Standard Landline/Corporate: 61234567
4. Continuous string with country code: 6593822338
5. Continuous string with plus: +6593822338
6. Standard format with plus & space: +65 9123 4567
7. Bracketed country code: (+65) 8123 4567
8. Dashed format: 9123-4567, +65 6123-4567
9. Toll-free/service lines: 1800 123 4567
10. STT transcript phonetic: "plus six five nine three eight two two three three eight"

## SG_ADDRESS (10 per letter A-Z, formatted as full realistic SG addresses
containing block + unit + postal code substrings for compositional tagging)

A: Blk 110 Admiralty Drive #02-14 S750110; Blk 22 Albert Street #14-22 S189970;
456 Alexandra Road #08-01 S119962; Blk 80 Aljunied Road #05-11 S389810;
18 Amoy Street #01-01 S069853; Blk 301 Anchorvale Link #11-54 S540301;
Blk 415 Ang Mo Kio Avenue 1 #03-120 S560415; 25 Ann Siang Road #02-00 S069705;
71 Arab Street #01-02 S199768; Blk 55 Ayer Rajah Crescent #04-18 S139949

B: 101 Balestier Road #02-15 S329678; Blk 4 Banda Street #08-22 S050004;
Blk 256 Bangkit Road #06-55 S670256; 18 Bartley Road #12-04 S539775;
505 Beach Road #33-01 S199588; Blk 12 Bedok North Avenue 1 #04-11 S460012;
80 Bencoolen Street #05-05 S189655; Blk 22 Bendemeer Road #09-12 S330022;
Blk 150 Bishan Street 11 #03-144 S570150; Blk 14 Boon Keng Road #02-12 S330014

C: 4 Cairnhill Road #11-02 S229649; Blk 1 Cantonment Road #05-16 S080001;
120 Cashew Road #04-22 S679685; 33 Ceylon Road #01-04 S429625;
400 Changi Road #02-10 S419847; 15 Cheong Chin Nam Road #01-01 S599739;
Blk 201 Choa Chu Kang Avenue 1 #12-34 S680201; Blk 35 Circuit Road #04-66 S370035;
Blk 311 Clementi Avenue 2 #08-112 S120311; Blk 50 Commonwealth Avenue #05-18 S149730

D: 8 Dalhousie Lane #02-01 S209677; Blk 85 Dawson Road #20-11 S141085;
12 Defu Lane 1 #01-14 S539487; 18 Dempsey Road #01-02 S249677;
Blk 110 Depot Road #06-12 S100110; 50 Desker Road #02-00 S209581;
112 Devonshire Road #04-15 S239877; Blk 22 Dover Crescent #03-55 S130022;
20 Duxton Hill #01-01 S089603; 350 Dunearn Road #08-11 S289655

E: 185 East Coast Road #02-14 S428891; Blk 112 Edgedale Plains #05-22 S820112;
Blk 115 Edgefield Plains #09-44 S820115; Blk 18 Eunos Crescent #11-55 S400018;
21 Ewe Boon Road #03-01 S259326; Blk 2 Everton Park #01-44 S081002;
Blk 625 Elias Road #06-112 S510625; 44 Eng Kong Terrace #01-01 S599017;
1 Esplanade Drive #02-05 S038981; 10 Exeter Road #14-01 S239732

F: Blk 4 Farrer Road #05-14 S260004; Blk 401 Fernvale Lane #12-33 S790401;
15 Fir Avenue #01-02 S279503; 50 Flora Drive #04-11 S506891;
20 Florence Road #02-05 S549525; 35 Foch Road #01-14 S209265;
11 Fort Canning Road #03-01 S179495; 120 Frankel Avenue #01-00 S458234;
Blk 8 French Road #06-18 S200008; 1 Fullerton Road #01-01 S049213

G: 18 Gali Batu Road #01-15 S678850; 50 Gambas Avenue #04-22 S756957;
Blk 110 Gangsa Road #08-44 S670110; 300 Geylang Road #02-14 S389343;
12 Gilstead Road #03-01 S309066; 8 Gloucester Road #05-11 S219460;
12 Gopeng Street #14-05 S078877; 25 Grange Road #08-08 S239699;
114 Guillemard Road #01-01 S399729; 15 Gul Circle #02-12 S629575

H: Blk 2 Haig Road #03-12 S430002; 10 Halifax Road #01-04 S229260;
2 Handy Road #05-01 S229233; Blk 22 Havelock Road #12-55 S160022;
18 Hazel Park Terrace #02-01 S678864; Blk 91 Henderson Road #08-112 S150091;
1 High Street #04-01 S179429; 112 Hillview Avenue #06-14 S669599;
Blk 15 Holland Avenue #03-44 S271015; Blk 110 Hougang Avenue 1 #08-22 S530110

I: 14 Idris Road #01-01 S329438; 2 Ilia Avenue #01-02 S140002;
40 Imbiah Road #02-05 S099701; 50 International Road #04-18 S629168;
5 Ipoh Lane #01-04 S438605; 20 Irrawaddy Road #11-22 S329562;
18 Irving Place #05-11 S369546; 10 Island Club Road #01-01 S578775;
2 Inggu Road #01-14 S759164; 5 Intan Road #02-05 S678821

J: 180 Jalan Besar #03-12 S208876; Blk 1 Jalan Bukit Merah #05-55 S150001;
25 Jalan Jurong Kechil #01-14 S598510; 250 Jalan Kayu #02-04 S799478;
Blk 18 Jalan Membina #12-22 S164018; 50 Jalan Sultan #04-15 S198974;
Blk 10 Jalan Toa Payoh #08-11 S310010; 114 Joo Chiat Road #01-01 S427406;
Blk 115 Jurong East Avenue 1 #06-44 S609787; Blk 411 Jurong West Street 41 #03-55 S640411

K: 15 Kallang Way #02-18 S349216; 80 Kampong Bahru Road #01-05 S169378;
12 Kapor Road #01-01 S208882; Blk 802 Keat Hong Close #14-22 S680802;
33 Keng Lee Road #03-11 S219277; 155 Keppel Road #08-15 S089058;
200 Kovan Road #04-02 S548174; 12 Kranji Way #01-12 S739428;
45 Kreta Ayer Road #02-04 S089005; 5 Kaki Bukit Avenue 1 #05-14 S417939

L: 110 Lavender Street #02-22 S338728; Blk 55 Lengkok Bahru #08-112 S151055;
12 Leonie Hill Road #14-01 S239194; 24 Lim Ah Pin Road #01-05 S547844;
15 Lim Tua Tow Road #02-01 S547746; 8 Lintang Square #01-01 S308708;
Blk 112 Lompang Road #06-55 S670112; Blk 305 Lorong Chuan #12-44 S550305;
1 Lorong Halus #01-14 S536557; 40 Loyang Avenue #03-22 S509058

M: 200 MacPherson Road #04-15 S348552; 15 Mandai Road #01-11 S779395;
8 Marina Boulevard #18-05 S018981; Blk 44 Marine Parade Road #12-55 S449266;
Blk 10 Marsiling Drive #05-12 S730010; 120 Marymount Road #02-18 S297705;
45 Maxwell Road #08-01 S069118; 88 Meyer Road #14-22 S437912;
115 Middle Road #03-11 S188977; 14 Mountbatten Road #01-04 S397992

N: 50 Nanyang Drive #04-15 S637553; 15 Nassim Road #02-01 S258385;
22 Nathan Road #05-11 S248744; 110 Neo Tiew Road #01-01 S719036;
200 New Bridge Road #08-12 S059419; 18 Newton Road #14-22 S307992;
10 Niven Road #01-04 S228357; 400 North Bridge Road #03-55 S188721;
100 North Buona Vista Road #06-01 S139345; 5 Novena Terrace #02-14 S307907

O: 68 Oasis Terrace #01-15 S820068; 11 Ocean Drive #08-11 S098396;
Blk 51 Old Airport Road #02-112 S390051; 18 Old Choa Chu Kang Road #01-14 S699806;
5 Olive Road #02-05 S298218; 20 Ophir Road #04-22 S188686;
15 Orange Grove Road #12-01 S258348; 1 Orchard Boulevard #05-11 S248649;
390 Orchard Road #14-04 S238871; 15 Outram Road #03-55 S169036

P: 40 Pagoda Street #01-01 S059199; Blk 41 Pandan Gardens #06-12 S600041;
250 Pasir Panjang Road #02-15 S118604; Blk 110 Pasir Ris Drive 1 #08-44 S510110;
100 Paya Lebar Road #04-11 S409005; 25 Penjuru Road #01-18 S609129;
Blk 202 Petir Road #12-33 S670202; 15 Pioneer Road #02-04 S628499;
Blk 115 Punggol Central #09-55 S820115; 12 Purvis Street #01-02 S188591

Q: 110 Queen Street #03-14 S188539; 1 Queensway #02-11 S149053;
15 Queen's Road #01-05 S260015; 8 Queen Astrid Park #01-01 S266810;
Blk 12 Queen's Close #06-22 S140012; Blk 20 Queen's Crescent #04-18 S140020;
5 Quayside Isle #01-11 S098375; 14 Quarry Way #02-04 S670014;
8 Queensborough Road #01-01 S759169; 22 Queen's Walk #03-05 S260022

R: 150 Race Course Road #02-12 S218597; Blk 1 Radin Mas #08-44 S090001;
1 Raffles Quay #14-01 S048583; 115 Rangoon Road #01-14 S218395;
Blk 85 Redhill Close #06-55 S150085; 10 Republic Boulevard #04-11 S038970;
25 Rifle Range Road #02-04 S588373; 350 River Valley Road #12-22 S238375;
80 Robinson Road #05-15 S068898; 1 Rochor Road #03-12 S180001

S: 15 Sembawang Road #01-11 S779075; Blk 120 Sengkang East Way #06-22 S540120;
200 Serangoon Road #02-14 S218069; 10 Shenton Way #14-05 S079117;
45 Siglap Road #01-01 S455856; Blk 110 Simei Street 1 #08-44 S520110;
22 Sin Ming Road #04-18 S575618; 111 Somerset Road #05-12 S238164;
180 South Bridge Road #02-11 S058729; 30 Stevens Road #03-04 S257840

T: Blk 120 Tampines Avenue 1 #06-55 S520120; 10 Tanah Merah Kechil Road #02-14 S466668;
50 Tanglin Road #08-22 S247911; 210 Tanjong Katong Road #01-11 S437004;
100 Tanjong Pagar Road #04-15 S088521; Blk 44 Teban Gardens Road #12-33 S600044;
120 Telok Ayer Street #01-02 S068589; 350 Thomson Road #03-12 S307684;
Blk 18 Tiong Bahru Road #06-22 S163018; Blk 102 Toa Payoh Lorong 1 #05-112 S310102

U: Blk 10 Ubi Avenue 1 #02-14 S408933; 50 Ulu Pandan Road #06-22 S596472;
120 Upper Aljunied Road #01-05 S367845; 400 Upper Bukit Timah Road #04-18 S588214;
15 Upper Changi Road #02-11 S467341; 34 Upper Cross Street #05-12 S058340;
18 Upper Dickson Road #01-01 S207477; 210 Upper East Coast Road #08-22 S455285;
114 Upper Paya Lebar Road #03-14 S534833; 350 Upper Serangoon Road #12-55 S534780

V: 10 Vanda Road #01-01 S287771; 5 Vaughan Road #02-04 S358488;
12 Venus Drive #01-11 S574299; 24 Veerasamy Road #03-15 S207328;
8 Vermont Road #02-01 S228224; 15 Vernon Park #01-02 S367812;
200 Victoria Street #05-22 S188021; 18 Victory Road #01-14 S506822;
4 Villa Grove #01-01 S456860; 1 Vista Exchange Green #06-11 S138617

W: 110 Waterloo Street #02-12 S187968; 5 Wayang Satu #01-04 S160005;
150 West Coast Road #05-15 S127367; 10 West Camp Road #01-11 S797660;
45 Westwood Avenue #02-14 S648356; Blk 20 Whampoa Drive #08-44 S320020;
12 Winstedt Road #01-01 S227971; Blk 110 Woodlands Avenue 1 #06-22 S730110;
18 Woodleigh Link #03-11 S368002; 5 Woking Road #01-12 S138694

X: 1 Xilin Avenue #02-14 S486798; 2 Xilin Avenue #03-22 S486798;
1 Xilin Link #01-01 S486221 (note: X is extremely limited in real SG road
names -- variations built from block/unit structure on existing roads)

Y: 250 Yio Chu Kang Road #02-15 S545690; Blk 110 Yishun Avenue 1 #05-22 S760110;
Blk 220 Yishun Avenue 2 #08-44 S760220; Blk 335 Yishun Avenue 3 #12-33 S760335;
Blk 450 Yishun Avenue 4 #04-18 S760450; Blk 555 Yishun Avenue 5 #03-14 S760555;
Blk 612 Yishun Avenue 6 #06-55 S760612; Blk 115 Yishun Ring Road #02-11 S760115;
Blk 122 Yishun Street 11 #01-04 S760122; 5 Yuan Ching Road #08-22 S618641

Z: 10 Zehnder Road #01-01 S117688; 110 Zion Road #02-15 S247771

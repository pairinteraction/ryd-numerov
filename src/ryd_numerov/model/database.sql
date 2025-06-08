BEGIN TRANSACTION;

--------------------------------------------------------
---------- Parameters for the model Potential ----------
--------------------------------------------------------
DROP TABLE IF EXISTS `model_potential`;
CREATE TABLE `model_potential` (`species` TEXT, `l` INT, `ac` REAL, `Z` INT, `a1` REAL, `a2` REAL, `a3` REAL, `a4` REAL, `rc` REAL);

-- Hydrogen atom
INSERT INTO `model_potential` VALUES ('H', 0, '0', 1, '0', '0', '0', '0', '0');
INSERT INTO `model_potential` VALUES ('H_textbook', 0, '0', 1, '0', '0', '0', '0', '0');

-- Helium ion
INSERT INTO `model_potential` VALUES ('He+', 0, '0', 1, '0', '0', '0', '0', '0');

-- Phys. Rev. A 49, 982 (1994)
INSERT INTO `model_potential` VALUES ('Li', 0, '0.1923', 3, '2.47718079', '1.84150932', '-0.02169712', '-0.11988362', '0.61340824');
INSERT INTO `model_potential` VALUES ('Li', 1, '0.1923', 3, '3.45414648', '2.55151080', '-0.21646561', '-0.06990078', '0.61566441');
INSERT INTO `model_potential` VALUES ('Li', 2, '0.1923', 3, '2.51909839', '2.43712450', '0.32505524', '0.10602430', '2.34126273');
INSERT INTO `model_potential` VALUES ('Li', 3, '0.1923', 3, '2.51909839', '2.43712450', '0.32505524', '0.10602430', '2.34126273');

-- Phys. Rev. A 49, 982 (1994)
INSERT INTO `model_potential` VALUES ('Na', 0, '0.9448', 11, '4.82223117', '2.45449865', '-1.12255048', '-1.42631393', '0.45489422');
INSERT INTO `model_potential` VALUES ('Na', 1, '0.9448', 11, '5.08382502', '2.18226881', '-1.19534623', '-1.03142861', '0.45798739');
INSERT INTO `model_potential` VALUES ('Na', 2, '0.9448', 11, '3.53324124', '2.48697936', '-0.75688448', '-1.27852357', '0.71875312');
INSERT INTO `model_potential` VALUES ('Na', 3, '0.9448', 11, '1.11056646', '1.05458759', '1.73203428', '-0.09265696', '28.6735059');

-- Phys. Rev. A 49, 982 (1994)
INSERT INTO `model_potential` VALUES ('K', 0, '5.3310', 19, '3.56079437', '1.83909642', '-1.74701102', '-1.03237313', '0.83167545');
INSERT INTO `model_potential` VALUES ('K', 1, '5.3310', 19, '3.65670429', '1.67520788', '-2.07416615', '-0.89030421', '0.85235381');
INSERT INTO `model_potential` VALUES ('K', 2, '5.3310', 19, '4.12713694', '1.79837462', '-1.69935174', '-0.98913582', '0.83216907');
INSERT INTO `model_potential` VALUES ('K', 3, '5.3310', 19, '1.42310446', '1.27861156', '4.77441476', '-0.94829262', '6.50294371');

-- Phys. Rev. A 49, 982 (1994)
INSERT INTO `model_potential` VALUES ('Rb', 0, '9.076', 37, '3.69628474', '1.64915255', '-9.86069196', '0.19579987', '1.66242117');
INSERT INTO `model_potential` VALUES ('Rb', 1, '9.076', 37, '4.44088978', '1.92828831', '-16.79597770', '-0.81633314', '1.50195124');
INSERT INTO `model_potential` VALUES ('Rb', 2, '9.076', 37, '3.78717363', '1.57027864', '-11.6558897', '0.52942835', '4.86851938');
INSERT INTO `model_potential` VALUES ('Rb', 3, '9.076', 37, '2.39848933', '1.76810544', '-12.0710678', '0.77256589', '4.79831327');

-- Phys. Rev. A 49, 982 (1994)
INSERT INTO `model_potential` VALUES ('Cs', 0, '15.6440', 55, '3.49546309', '1.47533800', '-9.72143084', '0.02629242', '1.92046930');
INSERT INTO `model_potential` VALUES ('Cs', 1, '15.6440', 55, '4.69366096', '1.71398344', '-24.65624280', '-0.09543125', '2.13383095');
INSERT INTO `model_potential` VALUES ('Cs', 2, '15.6440', 55, '4.32466196', '1.61365288', '-6.70128850', '-0.74095193', '0.93007296');
INSERT INTO `model_potential` VALUES ('Cs', 3, '15.6440', 55, '3.01048361', '1.40000001', '-3.20036138', '0.00034538', '1.99969677');

-- Own fit, ac: J. Phys. B 43 202001 (2010), singlet states -- TODO perform fit with ac=91.10

-- http://dollywood.itp.tuwien.ac.at/~shuhei/pub_Rydberg/PhysRevA.89.023426.pdf
INSERT INTO `model_potential` VALUES ('Sr88_singlet', 0, '7.5', 38, '3.36124', '1.3337', '0', '5.94337', '1.59');
INSERT INTO `model_potential` VALUES ('Sr88_singlet', 1, '7.5', 38, '3.28205', '1.24035', '0', '3.78861', '1.58');
INSERT INTO `model_potential` VALUES ('Sr88_singlet', 2, '7.5', 38, '2.155', '1.4545', '0', '4.5111', '1.57');
INSERT INTO `model_potential` VALUES ('Sr88_singlet', 3, '7.5', 38, '2.1547', '1.14099', '0', '2.1987', '1.56');

-- http://dollywood.itp.tuwien.ac.at/~shuhei/pub_Rydberg/PhysRevA.89.023426.pdf
-- TODO FIXME: currently we simply use the same parameters as for the singlet states
INSERT INTO `model_potential` VALUES ('Sr88_triplet', 0, '7.5', 38, '3.36124', '1.3337', '0', '5.94337', '1.59');
INSERT INTO `model_potential` VALUES ('Sr88_triplet', 1, '7.5', 38, '3.28205', '1.24035', '0', '3.78861', '1.58');
INSERT INTO `model_potential` VALUES ('Sr88_triplet', 2, '7.5', 38, '2.155', '1.4545', '0', '4.5111', '1.57');
INSERT INTO `model_potential` VALUES ('Sr88_triplet', 3, '7.5', 38, '2.1547', '1.14099', '0', '2.1987', '1.56');

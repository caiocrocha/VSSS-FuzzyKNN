// Implementa��o do k-NN (k-nearest neighbors algorithm)

// Código adaptado de https://github.com/marcoscastro/knn

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <set>
#include <map>
#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include "CSVIterator.h"

bool is_number(const std::string& s)
{
	std::string::const_iterator it = s.begin();
	while (it != s.end() && std::isdigit(*it)) ++it;
	return !s.empty() && it == s.end();
}

typedef struct
{
    float x, y;
    float theta = 0.0;
} Point;

class Vector
{
private:
    float x, y;

public:
    Vector(Point p1, Point p2)
    {
        this->x = p2.x - p1.x;
        this->y = p2.y - p1.y;
    }

    float getX() { return x; };
    float getY() { return y; };
    void setX(float x) { this->x = x; };
    void setY(float) { this->y = y; };
};

void generator(std::vector<Point> &robots, std::vector<Point> &ball)
{
    for(float i = 0; i < 180; i++)
    {
        for(float j = 0; j < 90; j++)
        {
            for(float k = -3.14159; k <= 3.14161; k+=1.0472) // float k = -pi; k <= pi; k+=pi/3
            {
                if(k > 3.14159)
                    k = 3.14159;
                robots.push_back(Point{i, j, k});
            }
            ball.push_back(Point{i, j});
        }
    }
    srand(time(NULL));
    shuffle(robots.begin(), robots.end(), std::default_random_engine(rand()));
    shuffle(ball.begin(), ball.end(), std::default_random_engine(rand()));
}

void write_points(std::vector<Point> &robots, std::vector<Point> &ball, 
                    std::string file_robots, std::string file_ball)
{
    int tam1 = robots.size();
    int tam2 = ball.size();
    std::ofstream file1, file2;
    file1.open(file_robots);
    file2.open(file_ball);
    file1 << "X,Y,Theta" << std::endl;
	file2 << "X,Y" << std::endl;
	for(int i = 0; i < tam1; i++)
	{
		file1 << robots[i].x << "," << robots[i].y << "," << robots[i].theta << std::endl;
        if(i < tam2)
            file2 << ball[i].x << "," << ball[i].y << std::endl;
        else
            file2.close();
	}
	file1.close();
}

using namespace std;

class Individuo
{
private:
    string classe;
    float x, y, theta;
    float df, cf, af;

public:
    Individuo(float x, float y, float theta)
    {
        this->x = x;
        this->y = y;
        this->theta = theta;
    };

    string getClasse() { return classe; };
    float getX() { return x; };
    float getY() { return y; };
    float getTheta() { return theta; };
    float getDF() { return df; };
    float getCF() { return cf; };
    float getAF() { return af; };
    
    void setFactors(float df, float cf, float af)
    {
        this->df = df;
        this->cf = cf;
        this->af = af;
    }

    void setClasse(string classe)
    {
        this->classe = classe;
    }

};

// fun��o que retorna a dist�ncia euclidiana entre 2 indiv�duos
float obterDistEuclidiana(Individuo ind1, Individuo ind2)
{
    /*
       a dist�ncia euclidiana � a raiz quadrada da soma das
       diferen�as dos valores dos atributos elevado ao quadrado
   */

    // ! desconsiderar theta
    float soma = pow((ind1.getX() - ind2.getX()), 2) +
                  pow((ind1.getY() - ind2.getY()), 2) +
                  pow((ind1.getDF() - ind2.getDF()), 2) +
                  pow((ind1.getCF() - ind2.getCF()), 2) +
                  pow((ind1.getAF() - ind2.getAF()), 2);

    return sqrt(soma);
}

void conta_classes(string classe, vector<int> &cont_classes)
{
    if (classe == "striker")
        cont_classes[0]++;
    else if (classe == "fake9")
        cont_classes[1]++;
    else if (classe == "wing")
        cont_classes[2]++;
    else if (classe == "midfield")
        cont_classes[3]++;
    else if (classe == "defender")
        cont_classes[4]++;
    else
        cont_classes[5]++;
}

// essa fun��o classifica uma nova amostra
string classificarAmostra(vector<Individuo> &individuos,
                          Individuo novo_exemplo, int K)
{
    // se o K for par decrementa
    if (K % 2 == 0)
    {
        K--;
        if (K <= 0)
            K = 1;
    }

    // obt�m o tamanho do vetor
    int tam_vet = individuos.size();

    /*
       set de pairs da dist�ncia de cada indiv�duo
       do conjunto de treinamento para o novo exemplo
       cada pair � composto pela dist�ncia e o �ndice
       do indiv�duo no vetor
   */
    set<pair<float, int>> dist_individuos;

    /*
       calcula-se a dist�ncia euclidiana do novo exemplo
       para cada amostra do conjunto de treinamento
   */
    for (int i = 0; i < tam_vet; i++)
    {
        float dist = obterDistEuclidiana(individuos[i], novo_exemplo);
        dist_individuos.insert(make_pair(dist, i));
    }
    /*
   para decidir a qual classe pertence o novo exemplo,
   basta verificar a classe mais frequente dos K
   vizinhos mais pr�ximos
   */
    set<pair<float, int>>::iterator it;

    /*
       o contador de striker estar� no �ndice 0,
       o contador de fake9 estar� no �ndice 1
       ....
   */
    vector<int> cont_classes(6);

    int contK = 0;

    for (it = dist_individuos.begin(); it != dist_individuos.end(); it++)
    {
        if (contK == K)
            break;

        string classe = individuos[it->second].getClasse();
        conta_classes(classe, cont_classes);

        contK++;
    }

    vector< pair <int,string> > sorted;
    sorted.push_back(make_pair(cont_classes[0], "striker"));
    sorted.push_back(make_pair(cont_classes[1], "fake9"));
    sorted.push_back(make_pair(cont_classes[2], "wing"));
    sorted.push_back(make_pair(cont_classes[3], "midfield"));
    sorted.push_back(make_pair(cont_classes[4], "defender"));
    sorted.push_back(make_pair(cont_classes[5], "goalkeeper"));

    sort(sorted.begin(), sorted.end());
    string classe_classificacao = sorted[5].second;

    return classe_classificacao;
}

float distancePoints(Point p1, Point p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

float angleVectors(Vector v1, Vector v2)
{
    return acos((v1.getX() * v2.getX() + v1.getY() * v2.getY()) / (sqrt(v1.getX() * v1.getX() + v1.getY() * v1.getY()) + sqrt(v2.getX() * v2.getX() + v2.getY() * v2.getY())));
}

float defenseFactor(Point robot, Point centroidAtk, Point centroidDef)
{
    float datk = distancePoints(robot, centroidAtk);
    float ddef = distancePoints(robot, centroidDef);
    return (ddef == 0 ? 0 : pow(2.7183, -0.6931 * datk / ddef));
}

float competitionFactor(Point robot, vector<Point> &enemies, Point ball)
{
    // Find closest enemy
    int closest = 0;
    float dcl = distancePoints(enemies[0], ball);
    for (int i = 0; i < 3; i++)
    {
        float d = distancePoints(enemies[i], ball);
        if (d < dcl)
        {
            closest = i;
            dcl = d;
        }
    }
    float dr = distancePoints(robot, ball);
    float de = distancePoints(enemies[closest], ball);
    return (de == 0 ? 0 : pow(2.7183, -0.6931 * dr / de));
}

float angleFactor(Point robot, Point ball, Point centroidAtk)
{
    Point origin = {0.0, 0.0};
    Point ux = {1.0, 0.0};
    Vector Vx(origin, ux);
    // angle between robot and ball
    float beta = angleVectors(Vector(ball, robot), Vx) - robot.theta;
    // angle between robot and centroidAtk
    float gamma = angleVectors(Vector(centroidAtk, robot), Vx) - robot.theta;
    float k1, k2;
    if (beta >= -90 && beta <= 90)
        k1 = (90 - abs(beta)) / 90;
    else
        k1 = (-90 + abs(beta)) / 90;
    if (gamma >= -90 && gamma <= 90)
        k2 = (90 - abs(gamma)) / 90;
    else
        k2 = (-90 + abs(gamma)) / 90;
    return 0.7 * k1 + 0.3 * k2;
}

void preencher(string file_robots, string file_ball, int tam, 
                    vector<Individuo> &individuos, vector<Point> &enemies, vector<Point> &ball)
{
    ifstream frobots(file_robots);
    CSVIterator loop(frobots);
	for(int i = 0; i <= tam && loop != CSVIterator(); ++loop, i++)
	{
		if(is_number((*loop)[0]))
		{
			float x = stod((*loop)[0]);
			float y = stod((*loop)[1]);
            float theta = stod((*loop)[2]);
			individuos.push_back(Individuo(x, y, theta));
		}
	}

    for(int i = 0; i <= tam && loop != CSVIterator(); ++loop, i++)
	{
		if(is_number((*loop)[0]))
		{
			float x = stod((*loop)[0]);
			float y = stod((*loop)[1]);
			enemies.push_back(Point{x, y});
		}
	}
    frobots.close();

    ifstream fball(file_ball);
    CSVIterator loop2(fball);
    for(int i = 0; i <= tam && loop2 != CSVIterator(); ++loop2, i++)
    {
        if(is_number((*loop2)[0]))
		{
			float x = stod((*loop2)[0]);
			float y = stod((*loop2)[1]);
			ball.push_back(Point{x, y});
		}
    }
    fball.close();
}

void escreverCSV(string arquivo, vector<Individuo> &individuos)
{
    ofstream file_out;
	file_out.open(arquivo);
	file_out << "X,Y,Theta" << endl;
	for(int i = 0; i < individuos.size(); i++)
	{
		file_out << individuos[i].getX() << "," << individuos[i].getY() << "," << individuos[i].getTheta() << endl;
	}
	file_out.close();
}

void embaralhaIndividuos(vector<Individuo> &individuos)
{
    srand(time(NULL));
    shuffle(individuos.begin(), individuos.end(), default_random_engine(rand()));
}

void embaralhaPontos(vector<Point> &vetor)
{
    srand(time(NULL));
    shuffle(vetor.begin(), vetor.end(), default_random_engine(rand()));
}

string atribui_classe(float df, float cf, float af)
{
    string classe;
    if(df > 0.66)
    {
        if(cf < 0.33)
            classe = "fake9";
        else if(af < 0.96)
            classe = "wing";
        else
            classe = "striker";
    }
    else if(df > 0.33)
    {
        if(cf > 0.66)
            classe = "striker";
        else if(cf > 0.33)
            classe = "fake9";
        else
            classe = "midfield";
    }
    else
    {
        if(cf < 0.33)
            classe = "goalkeeper";
        else
            classe = "defender";
    }

    return classe;
}

int main(int argc, char *argv[])
{
    string file_robots, file_ball;
    file_robots = "table.csv";
    file_ball = "ball.csv";
    vector<Point> ball;

    /*
    // gera novas posições aleatórias para os arquivos com as coordenadas
    vector<Point> robots;
    generator(robots, ball);
    write_points(robots, ball, file_robots, file_ball);
    ball.clear();
    */

    vector<Individuo> individuos;
    vector<Point> enemies;
    int tam = 162;
    Point centroidAtk = {175, 45};
    Point centroidDef = {5, 45};
   
    preencher(file_robots, file_ball, tam, individuos, enemies, ball);

    /*
    embaralhaIndividuos(individuos);
    embaralhaPontos(enemies);
    embaralhaPontos(ball);
    */
   
    /*
       o K � a quantidade de vizinhos que ser�o
       levados em conta para classifica��o de um
       novo dado, � recomend�vel que seja �mpar
       para que n�o possa haver empate
   */
    int K = 3;

    // tamanho do conjunto de dados de treinamento
    int tam_treinamento = ceil(0.7*tam);

    /*
       o processo de treinamento consiste em apenas
       armazenar o conjunto de dados de treinamento
   */

    int acertos = 0;
    int tam_testes = tam - tam_treinamento;

    // processo de treinamento
    for (int i = 0, j = 0; i < tam; i++, j+=3)
    {
        int k = j % tam;
        Point robot = {individuos[i].getX(), individuos[i].getY(), individuos[i].getTheta()};
        vector<Point> enemiesK {enemies[k], enemies[k+1], enemies[k+2]};
        float df = defenseFactor(robot, centroidAtk, centroidDef);
        float cf = competitionFactor(robot, enemiesK, ball[i]);
        float af = angleFactor(robot, ball[i], centroidAtk);
        individuos[i].setFactors(df, cf, af);

        string classe = atribui_classe(df, cf, af);
        individuos[i].setClasse(classe);
    }

    vector<int> cont_classes(6, 0);
    vector<int> cont_classes_obtidas(6, 0);

    // processo de classifica��o
    for (int i = tam_treinamento; i < tam; i++)
    {
        string classe = individuos[i].getClasse();
        string classe_obtida = classificarAmostra(individuos, individuos[i], K);
        
        cout << "Classe esperada: " << classe << endl;
        cout << "Classe obtida: " << classe_obtida << endl << endl;

        conta_classes(classe, cont_classes);
        conta_classes(classe_obtida, cont_classes_obtidas);

        if (classe == classe_obtida)
            acertos++;
    }

    cout << "striker: " << cont_classes[0] << " esperados e " << cont_classes_obtidas[0] << " obtidas" << endl;
    cout << "fake9: " << cont_classes[1] << " esperados e " << cont_classes_obtidas[1] << " obtidas" << endl;
    cout << "wing: " << cont_classes[2] << " esperados e " << cont_classes_obtidas[2] << " obtidas" << endl;
    cout << "midfield: " << cont_classes[3] << " esperados e " << cont_classes_obtidas[3] << " obtidas" << endl;
    cout << "defender: " << cont_classes[4] << " esperados e " << cont_classes_obtidas[4] << " obtidas" << endl;
    cout << "goalkeeper: " << cont_classes[5] << " esperados e " << cont_classes_obtidas[5] << " obtidas" << endl;

    cout << acertos << " acertos de um total de " << tam_testes << " testes (" 
        << acertos/float(tam_testes)*100 << "%)" << endl;

    // escreverCSV(arquivo, individuos);

    return 0;
}
